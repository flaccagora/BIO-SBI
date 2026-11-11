"""Sampler coordinates for ``exact_bh`` that straighten the lifetime / self-renewal ridge.

``exact_bh`` samples ``u = logit`` of the unit-scaled theta coordinates (stick-breaking
sticks, log mean lifetimes). There the posterior lies along a curved ridge: the data fix
the clone growth rate and the A:B composition, which several combinations of outcome
probabilities and lifetimes reproduce (more self-renewal ``A.div_AA`` with a longer A
lifetime; the B lifetime with B's row), and probabilities the data allow to be ~0
(``A.div_AB``, ``A.persist``) have exponential tails towards ``u -> -inf``.
:class:`Reparam` is a bijection ``v -> u`` of ``R^d`` with exact log Jacobian; the
sampler targets ``log p(u(v)) + log |du/dv|``, so the posterior (and the prior) of ``u``
is unchanged. ``spec`` is one of :data:`SPECS`:

* ``stable`` — shear both free log mean lifetimes (columns ``k`` = A, B):

      v_k = u_k - g_k(sticks),     u_k = v_k + g_k(sticks),     |du/dv| = 1,

  with ``g_k = logit(clip((l*_k - low_k) / W_k, CLIP, 1 - CLIP))``, ``l*`` = the log
  lifetimes at which the Malthusian parameter (asymptotic exponential growth rate)
  equals ``alpha0`` *and* the stable ratio of living B to A cells equals ``r0`` (both at
  the anchor, the posterior mode), given the outcome probabilities. With ``M`` the mean
  offspring matrix and ``L_j(alpha) = (1 + alpha tau_j / m_j)^-m_j`` the Laplace
  transform of the ``Gamma(m_j, tau_j / m_j)`` lifetime, the Malthusian parameter solves
  ``rho(diag(L(alpha)) M) = 1`` (two types: ``det(I - diag(L) M) = 0``) and the stable
  ratio is ``r = L_A M_AB (1 - L_B) / ((1 - L_B M_BB) (1 - L_A))`` (left Perron vector
  of ``diag(L) M`` for the births, ``(1 - L_j) / alpha`` discounted mean life);
  :func:`stable_lifetimes` solves the pair for ``(L_A, L_B)`` (closed form in ``L_A``,
  bisection in ``L_B``). The posterior surface of constant growth and composition becomes
  ``v_A = v_B = const``. Where no solution exists, ``g`` = the anchor's value.
* ``growth`` — shear only the A lifetime onto constant ``alpha0`` (closed form, given B's
  lifetime: :func:`ridge_lifetime`). Weaker: B's lifetime ridge stays curved.
* ``probit`` — stick coordinates ``v = probit(sigmoid(u))`` (log mean lifetimes stay
  logit: uniform prior): Gaussian instead of exponential tails for probabilities piling
  up at 0 or 1 (``log |du/dv| = log phi(v) - log Phi(v) - log Phi(-v)`` per stick);
  combined as ``stable+probit`` / ``growth+probit`` (probit first, then the shear, which
  depends on the sticks only through ``u``).

Any measurable ``g`` of the unsheared coordinates gives a bijection with unit Jacobian:
the ridge solution, the clip (``g`` bounded and continuous where the ridge leaves the
prior box) and the fallback only affect mixing, never the target. Linear whitening is
not needed: the samplers' random walk / independence proposals are affine invariant;
what the shear buys is a posterior closer to Gaussian, so the independence (t) proposal
is accepted more often (validation design: 0.07 -> 0.18).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, log_ndtr, ndtri

#: Clip of the unit-scaled ridge lifetime in the shear (``|g| <= logit(1 - CLIP)`` = 4.6).
CLIP = 0.01
#: Recognised specs.
SPECS = ("none", "growth", "stable", "probit", "growth+probit", "stable+probit")


# ---------------------------------------------------------------------- growth theory
def gamma_laplace(alpha, tau, shape):
    """Laplace transform ``E exp(-alpha T)`` of ``T ~ Gamma(shape, tau / shape)``.

    ``nan`` where ``1 + alpha tau / shape <= 0`` (the transform diverges).
    """
    z = 1.0 + np.asarray(alpha, float) * np.asarray(tau, float) / shape
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(z > 0, np.exp(-shape * np.log(np.where(z > 0, z, 1.0))), np.nan)


def _rho2(K: np.ndarray) -> np.ndarray:
    """Spectral radius (Perron root) of nonnegative ``(..., 2, 2)`` matrices."""
    tr = K[..., 0, 0] + K[..., 1, 1]
    det = K[..., 0, 0] * K[..., 1, 1] - K[..., 0, 1] * K[..., 1, 0]
    return 0.5 * (tr + np.sqrt(np.maximum(tr**2 - 4 * det, 0.0)))


def malthusian(M: np.ndarray, tau: np.ndarray, shape: np.ndarray, iters: int = 100
               ) -> np.ndarray:  # fmt: skip
    """Malthusian parameter of a multi-type Bellman-Harris process with Gamma lifetimes.

    Root ``alpha`` of ``rho(diag(L(alpha)) M) = 1`` (``L`` = :func:`gamma_laplace` per
    type); ``rho`` decreases in ``alpha``. Vectorised over leading axes: ``M (..., k, k)``,
    ``tau (..., k)``, ``shape (k,)``. ``nan`` where the process cannot grow or shrink
    (``rho(M) = 0``). Bisection (any number of types; closed-form eigenvalues for two).
    """
    M = np.asarray(M, float)
    tau = np.asarray(tau, float)
    shape = np.broadcast_to(np.asarray(shape, float), tau.shape)
    k = M.shape[-1]

    def rho(alpha):
        L = gamma_laplace(alpha[..., None], tau, shape)
        K = L[..., :, None] * M
        if k == 2:
            return _rho2(K)
        return np.max(np.abs(np.linalg.eigvals(K)), axis=-1)

    # bracket: alpha > -min(shape / tau) (transform finite); rho -> inf at the left end
    lo = -np.min(shape / tau, axis=-1) * (1 - 1e-9)
    hi = np.ones(lo.shape)
    for _ in range(200):  # grow hi until rho(hi) < 1
        bad = rho(hi) >= 1
        if not bad.any():
            break
        hi = np.where(bad, hi * 2, hi)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        up = rho(mid) >= 1
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    out = 0.5 * (lo + hi)
    return np.where(_rho2(M) > 0 if k == 2 else True, out, np.nan)


def ridge_lifetime(M: np.ndarray, tau_other: np.ndarray, shape_k: float, shape_other: float,
                   alpha0: float, k: int = 0) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Mean lifetime of type ``k`` giving Malthusian parameter ``alpha0`` (two types).

    Returns ``(log tau_k*, side)``: ``side`` is 0 where the root exists, else ``+1``
    (``log tau_k* = +inf``) where the rest alone grows faster than ``alpha0`` (for
    ``alpha0 > 0``), ``-1`` (``-inf``) otherwise (a direction for the shear, not an
    optimum: ``alpha`` need not be monotone in ``tau_k`` when type ``k`` is subcritical).
    """
    o = 1 - k
    a, b = M[..., k, k], M[..., k, o]
    c, d = M[..., o, k], M[..., o, o]
    y = gamma_laplace(alpha0, tau_other, shape_other)
    one_m = 1.0 - y * d
    den = a * one_m + y * b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        x = one_m / den
        z = np.power(x, -1.0 / shape_k) - 1.0  # alpha0 tau / m
        tau = shape_k * z / alpha0
    ok = (one_m > 0) & (den > 0) & np.isfinite(tau) & (tau > 0)
    # no root: the rest grows too fast (the other block alone reaches alpha0, or for
    # alpha0 < 0 even tau_k -> 0 is too fast) or too slow for any positive tau_k
    too_fast = (one_m <= 0) | ((alpha0 < 0) & ~ok)
    up = np.where(too_fast, alpha0 > 0, alpha0 < 0)  # tau_k* -> inf (else -> 0)
    side = np.where(ok, 0, np.where(up, 1, -1))
    log_tau = np.where(ok, np.log(np.where(ok, tau, 1.0)), np.where(up, np.inf, -np.inf))
    return log_tau, side


def stable_ratio(M: np.ndarray, tau: np.ndarray, shape: np.ndarray, alpha: np.ndarray
                 ) -> np.ndarray:  # fmt: skip
    """Stable ratio of living type-1 to type-0 cells (two types, Malthusian ``alpha``).

    In the stable regime type-``j`` cells are born at rate ``beta_j e^(alpha t)`` with
    ``beta`` the left Perron vector of ``K = diag(L(alpha)) M`` and live
    ``(1 - L_j) / alpha`` on average (discounted), so with ``x = L_0``, ``y = L_1``:
    ``r = x M_01 (1 - y) / ((1 - y M_11) (1 - x))``.
    """
    x = gamma_laplace(alpha, tau[..., 0], shape[0])
    y = gamma_laplace(alpha, tau[..., 1], shape[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return x * M[..., 0, 1] * (1 - y) / ((1 - y * M[..., 1, 1]) * (1 - x))


def stable_lifetimes(M: np.ndarray, shape: np.ndarray, alpha0: float, r0: float,
                     iters: int = 80) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Both mean lifetimes giving Malthusian parameter ``alpha0`` and stable ratio ``r0``.

    For fixed outcome probabilities (``M (n, 2, 2)``) and ``alpha0 > 0`` the unknowns are
    ``x = L_0(alpha0)``, ``y = L_1(alpha0)`` in ``(0, 1)``: the growth equation gives
    ``x(y) = (1 - y d) / (a (1 - y d) + y b c)`` and ``y`` solves
    ``r0 (1 - y d) (1 - x(y)) = x(y) b (1 - y)`` (bisection on ``y < min(1, 1/d)``).
    Returns ``(log tau (n, 2), ok)``; ``ok`` is False where no root with ``x, y`` in
    ``(0, 1)`` exists (``log tau`` is then ``nan``).
    """
    a, b, c, d = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]

    def x_of(y):
        om = 1 - y * d
        with np.errstate(divide="ignore", invalid="ignore"):
            return om / (a * om + y * b * c)

    def F(y):
        x = x_of(y)
        return r0 * (1 - y * d) * (1 - x) - x * b * (1 - y)

    eps = 1e-12
    lo = np.full(len(M), eps)
    hi = np.minimum(1.0, 1.0 / np.maximum(d, eps)) - eps
    f_lo, f_hi = F(lo), F(hi)
    ok = np.isfinite(f_lo) & np.isfinite(f_hi) & (np.sign(f_lo) != np.sign(f_hi))
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        f_mid = F(mid)
        left = np.sign(f_mid) == np.sign(f_lo)
        lo, f_lo = np.where(left, mid, lo), np.where(left, f_mid, f_lo)
        hi = np.where(left, hi, mid)
    y = 0.5 * (lo + hi)
    x = x_of(y)
    ok &= (x > 0) & (x < 1) & (y > 0) & (y < 1)
    out = np.full((len(M), 2), np.nan)
    for j, L in enumerate((x, y)):
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = shape[j] * (np.power(L, -1.0 / shape[j]) - 1.0) / alpha0
            out[:, j] = np.where(ok, np.log(np.where(ok & (tau > 0), tau, 1.0)), np.nan)
    return out, ok


# ---------------------------------------------------------------------- bijection
class Reparam:
    """Bijection ``v -> u`` for an :class:`~bio_sbi.inference.exact_bh.BHLikelihood`.

    Args:
        lik: The likelihood (theta layout, prior box, law, lifetime shapes).
        spec: One of :data:`SPECS`.
        anchor_u: Point (``u`` coordinates) fixing ``alpha0`` (and ``r0``), usually the
            posterior mode.
        shear_type: ``growth``: type whose log mean lifetime is sheared (default A).
    """

    def __init__(self, lik, spec: str = "stable", anchor_u: np.ndarray | None = None,
                 shear_type: int = 0) -> None:  # fmt: skip
        spec = str(spec or "none")
        if spec not in SPECS:
            raise ValueError(f"exact_bh: unknown reparam {spec!r} ({' | '.join(SPECS)})")
        self.spec = spec
        self.lik = lik
        self.dim = lik.dim
        self.stick = lik.kind == 0
        self.probit = "probit" in spec
        self.mode = next((m for m in ("growth", "stable") if m in spec), None)
        self.k = int(shear_type)
        self.alpha0 = self.r0 = float("nan")
        self.cols: list[int] = []
        if self.mode is None:
            return
        sim = lik.sim
        types = list(sim.types)
        if len(types) != 2:
            raise ValueError(f"reparam={spec} needs two cell types")
        n_sticks = int(self.stick.sum())
        self.idx = {}  # type index -> theta column of its log mean lifetime (if free)
        for j, sc in enumerate(lik.space.free_scalars):
            self.idx[types.index(sc.name.split(".")[0])] = n_sticks + j
        need = [self.k] if self.mode == "growth" else [0, 1]
        for t in need:
            if t not in self.idx:
                raise ValueError(f"reparam={spec} needs a free {types[t]}.mean_lifetime")
        self.types = need
        self.cols = [self.idx[t] for t in need]
        self.shape = np.asarray(sim.shape, float)
        self.offspring = np.asarray(sim.law.offspring, float)
        self.row_slices = [lik.space.row_slice(t) for t in types]
        self.n_out = [int(n) for n in sim.law.n_outcomes]
        anchor_u = np.atleast_2d(np.zeros(self.dim) if anchor_u is None
                                 else np.asarray(anchor_u, float))  # fmt: skip
        M, tau = self._mean_and_tau(anchor_u)
        self.alpha0 = float(malthusian(M, tau, self.shape)[0])
        if not np.isfinite(self.alpha0) or abs(self.alpha0) < 1e-6:
            raise ValueError(f"reparam={spec}: Malthusian parameter at the anchor is "
                             f"{self.alpha0} (needs a growing or shrinking clone)")  # fmt: skip
        if self.mode == "stable":
            if self.alpha0 <= 0:
                raise ValueError(f"reparam={spec} needs a growing clone at the anchor "
                                 f"(alpha0 = {self.alpha0})")  # fmt: skip
            self.r0 = float(stable_ratio(M, tau, self.shape, self.alpha0)[0])
            if not (np.isfinite(self.r0) and self.r0 > 0):
                raise ValueError(f"reparam={spec}: stable type ratio at the anchor is {self.r0}")
        self.low = lik.low[self.cols].astype(float)
        self.width = lik.width[self.cols].astype(float)
        self.s_c = expit(anchor_u[0, self.cols])  # anchor (used where no ridge point exists)

    # ---------------------------------------------------------------- model quantities
    def _mean_and_tau(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean matrices ``(n, 2, 2)`` and mean lifetimes ``(n, 2)`` at rows of ``U``."""
        lik = self.lik
        theta = lik.theta_from_u(U)
        phys = lik.space.decode(theta)
        n = len(U)
        P = np.zeros((n,) + self.offspring.shape[:2])
        for i, sl in enumerate(self.row_slices):
            P[:, i, : self.n_out[i]] = np.clip(phys[:, sl], 0, None)
        M = np.einsum("nko,koj->nkj", P, self.offspring)
        tau = np.empty((n, 2))
        for i in range(2):
            if i in self.idx:
                tau[:, i] = np.exp(theta[:, self.idx[i]])
            else:  # fixed lifetime: same for every row
                tau[:, i] = lik.sim.mean_lifetimes(phys[0])[i]
        return M, tau

    def alpha(self, U: np.ndarray) -> np.ndarray:
        """Malthusian parameter (per hour) at rows of ``U`` (``u`` coordinates)."""
        M, tau = self._mean_and_tau(np.atleast_2d(U))
        return malthusian(M, tau, self.shape)

    def ratio(self, U: np.ndarray) -> np.ndarray:
        """Stable ratio of living B to A cells at rows of ``U`` (at their own ``alpha``)."""
        M, tau = self._mean_and_tau(np.atleast_2d(U))
        return stable_ratio(M, tau, self.shape, malthusian(M, tau, self.shape))

    def ridge_log_lifetimes(self, U: np.ndarray) -> np.ndarray:
        """Return the ``(n, len(cols))`` log lifetimes on the ridge.

        The sheared coordinates of ``U`` are ignored. ``growth``: +-inf where no lifetime
        reaches ``alpha0``; ``stable``: ``nan`` where no root exists.
        """
        M, tau = self._mean_and_tau(np.atleast_2d(U))
        if self.mode == "growth":
            o = 1 - self.k
            lt, _ = ridge_lifetime(M, tau[:, o], self.shape[self.k], self.shape[o],
                                   self.alpha0, self.k)  # fmt: skip
            return lt[:, None]
        return stable_lifetimes(M, self.shape, self.alpha0, self.r0)[0]

    def shear(self, U: np.ndarray) -> np.ndarray:
        """Offsets ``g`` (logit units, ``(n, len(cols))``) of the sheared coordinates.

        ``g = logit(clip((l* - low) / W, CLIP, 1 - CLIP))``: exactly the ridge inside the
        prior box, bounded and continuous where the ridge leaves it (``l* = +-inf``
        included). Depends only on the coordinates that are not sheared. No ridge point
        (``nan``, ``stable`` only): the anchor's value.
        """
        lt = self.ridge_log_lifetimes(U)
        with np.errstate(invalid="ignore"):
            s = np.clip((lt - self.low) / self.width, CLIP, 1 - CLIP)
        s = np.where(np.isnan(s), self.s_c, s)
        return np.log(s) - np.log1p(-s)

    # ---------------------------------------------------------------- bijection
    def to_u(self, V: np.ndarray) -> np.ndarray:
        """``u(v)`` (rows)."""
        V = np.atleast_2d(np.asarray(V, float))
        U = V.copy()
        if self.probit:
            Vs = V[:, self.stick]
            U[:, self.stick] = log_ndtr(Vs) - log_ndtr(-Vs)
        if self.cols:
            U[:, self.cols] = V[:, self.cols] + self.shear(U)
        return U

    def to_v(self, U: np.ndarray) -> np.ndarray:
        """``v(u)`` (rows); inverse of :meth:`to_u`."""
        U = np.atleast_2d(np.asarray(U, float))
        V = U.copy()
        if self.cols:
            V[:, self.cols] = U[:, self.cols] - self.shear(U)
        if self.probit:
            Us = U[:, self.stick]
            # probit(sigmoid(u)) via the smaller tail (no rounding of sigmoid to 1)
            V[:, self.stick] = np.where(Us <= 0, ndtri(expit(np.minimum(Us, 0))),
                                        -ndtri(expit(-np.maximum(Us, 0))))  # fmt: skip
        return V

    def log_jac(self, V: np.ndarray) -> np.ndarray:
        """``log |det du/dv|`` (rows); the shears contribute 0."""
        V = np.atleast_2d(np.asarray(V, float))
        if not self.probit:
            return np.zeros(len(V))
        Vs = V[:, self.stick]
        return (-0.5 * Vs**2 - 0.5 * math.log(2 * math.pi) - log_ndtr(Vs)
                - log_ndtr(-Vs)).sum(axis=1)  # fmt: skip

    def info(self) -> dict:
        """Return the settings for the diagnostics."""
        names = self.lik.space.theta_names
        return {"spec": self.spec, "alpha0": self.alpha0, "stable_ratio0": self.r0,
                "sheared": [names[c] for c in self.cols]}  # fmt: skip
