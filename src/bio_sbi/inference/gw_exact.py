"""Exact endpoint law of a two-type Galton-Watson clone, with exact derivatives.

The probability generating function (pgf) of the counts ``Z_n = (A, B)`` at generation
``n`` of a clone is the ``n``-fold iterate of the offspring pgfs,

    F_n = f(F_{n-1}),   f_k(x, y) = sum_o P[k, o] x^{a_ko} y^{b_ko},

started at the *observation* pgf ``F_0`` (identity for a perfect assay), and mixed over
the founder: ``G(x, y) = sum_m pi_m (sum_k phi_k F_n,k(x, y))^m``. The coefficients
``p[a, b] = P(Z_n = (a, b))`` follow from evaluating ``G`` on an ``Lx x Ly`` grid of
roots of unity and taking a 2-D inverse DFT. This is exact (to round-off) as long as
the law puts no mass on counts ``>= L``; any mass beyond wraps around ("aliasing"). To
make the wrapped mass harmless the grid is evaluated at radius ``r < 1`` (``z = r w``)
and the coefficients are rescaled by ``r^-(a + b)``: mass at ``a + Lx`` is then damped
by ``r^Lx`` relative to mass at ``a`` (see :func:`grid_spec`).

Derivatives with respect to the inference coordinates are propagated in forward mode:
for every direction ``d``, ``dF_n = (d_P f)[dP_d](F_{n-1}) + J_f(F_{n-1}) dF_{n-1}``.

Observation (capture ``c_k`` per true type, confusion ``C[k, j]`` from type to label)
enters through the pgf of one endpoint cell: ``F_0,k(x, y) = 1 - c_k + c_k (C[k,0] x +
C[k,1] y)``. Detected-clone selection conditions on ``Z != (0, 0)``; its probability
``1 - G(0, 0)`` is computed in complement form (no cancellation) by
:func:`detection_and_grad`.

Outcome names never matter here: the offspring table (daughters of every outcome) is
taken from :class:`~bio_sbi.simulators.law.OffspringLaw`, the same table the
simulators use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft
from numba import njit


# ---------------------------------------------------------------------- numba kernel
@njit(cache=True, nogil=True)
def _pgf_grid(  # noqa: D417 (grouped argument descriptions)
    P, dP, rows, off, n_out, gens, zx, zy, F0c, F0x, F0y, phi, sizes, pis, out, dout
):
    """Evaluate ``G`` and its ``D`` directional derivatives on the grid ``zx x zy``.

    Args:
        P: ``(2, O)`` outcome probabilities.
        dP: ``(D, O)`` derivative of the *one* row each direction moves.
        rows: ``(D,)`` type row moved by each direction.
        off: ``(2, O, 2)`` daughters per outcome.
        n_out: ``(2,)`` number of outcomes per type.
        gens: Generations.
        zx, zy: Grid points (complex) along the A and B axes.
        F0c, F0x, F0y: ``(2,)`` observation pgf ``F_0,k = F0c[k] + F0x[k] x + F0y[k] y``.
        phi: ``(2,)`` founder-type probabilities.
        sizes, pis: Founder sizes and their probabilities.
        out: ``(len(zx), len(zy))`` complex output.
        dout: ``(D, len(zx), len(zy))`` complex output (``D`` may be 0).
    """
    D = dP.shape[0]
    n_cols = P.shape[1]
    maxp = 0
    for k in range(2):
        for o in range(n_out[k]):
            for j in range(2):
                if off[k, o, j] > maxp:
                    maxp = off[k, o, j]
    dA = np.zeros(D, np.complex128)
    dB = np.zeros(D, np.complex128)
    nA = np.zeros(D, np.complex128)
    nB = np.zeros(D, np.complex128)
    powA = np.zeros(maxp + 1, np.complex128)
    powB = np.zeros(maxp + 1, np.complex128)
    mono = np.zeros((2, n_cols), np.complex128)
    for i in range(zx.shape[0]):
        x = zx[i]
        for j in range(zy.shape[0]):
            y = zy[j]
            FA = F0c[0] + F0x[0] * x + F0y[0] * y
            FB = F0c[1] + F0x[1] * x + F0y[1] * y
            for d in range(D):
                dA[d] = 0.0
                dB[d] = 0.0
            for _ in range(gens):
                powA[0] = 1.0
                powB[0] = 1.0
                for e in range(1, maxp + 1):
                    powA[e] = powA[e - 1] * FA
                    powB[e] = powB[e - 1] * FB
                fv0 = 0.0j
                fv1 = 0.0j
                fx0 = 0.0j
                fy0 = 0.0j
                fx1 = 0.0j
                fy1 = 0.0j
                for k in range(2):
                    for o in range(n_out[k]):
                        a = off[k, o, 0]
                        b = off[k, o, 1]
                        m = powA[a] * powB[b]
                        mono[k, o] = m
                        p = P[k, o]
                        if p == 0.0:
                            continue
                        gx = a * powA[a - 1] * powB[b] if a > 0 else 0.0j
                        gy = b * powA[a] * powB[b - 1] if b > 0 else 0.0j
                        if k == 0:
                            fv0 += p * m
                            fx0 += p * gx
                            fy0 += p * gy
                        else:
                            fv1 += p * m
                            fx1 += p * gx
                            fy1 += p * gy
                for d in range(D):
                    t = 0.0j
                    r = rows[d]
                    for o in range(n_out[r]):
                        t += dP[d, o] * mono[r, o]
                    nA[d] = fx0 * dA[d] + fy0 * dB[d]
                    nB[d] = fx1 * dA[d] + fy1 * dB[d]
                    if r == 0:
                        nA[d] += t
                    else:
                        nB[d] += t
                for d in range(D):
                    dA[d] = nA[d]
                    dB[d] = nB[d]
                FA = fv0
                FB = fv1
            S = phi[0] * FA + phi[1] * FB
            G = 0.0j
            dG = 0.0j
            for s in range(sizes.shape[0]):
                m = sizes[s]
                G += pis[s] * S**m
                dG += pis[s] * m * S ** (m - 1)
            out[i, j] = G
            for d in range(D):
                dout[d, i, j] = dG * (phi[0] * dA[d] + phi[1] * dB[d])


# ---------------------------------------------------------------------- model spec
@dataclass(frozen=True)
class GWModel:
    """Everything the exact law needs besides the outcome probabilities.

    Attributes:
        offspring: ``(2, O, 2)`` daughters per outcome (from the law).
        n_out: ``(2,)`` outcomes per type.
        capture: ``(2,)`` capture probability per true type.
        confusion: ``(2, 2)`` true type -> label probabilities.
        founder: ``(2,)`` founder-type probabilities (fixed).
        sizes: Founder sizes.
        size_probs: Their probabilities (fixed).
        detected: Condition on ``Z != 0`` (``selection=detected``).
    """

    offspring: np.ndarray
    n_out: np.ndarray
    capture: np.ndarray
    confusion: np.ndarray
    founder: np.ndarray
    sizes: np.ndarray
    size_probs: np.ndarray
    detected: bool = True

    @property
    def f0(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coefficients of the per-cell observation pgf ``c0 + cx x + cy y``."""
        c = self.capture
        return 1.0 - c, c * self.confusion[:, 0], c * self.confusion[:, 1]


@dataclass(frozen=True)
class GridSpec:
    """FFT grid of one horizon: sizes and damping radii (see :func:`grid_spec`)."""

    Lx: int
    Ly: int
    rx: float
    ry: float

    @property
    def points(self) -> int:
        """Grid points evaluated (half the B axis: the law is real)."""
        return self.Lx * (self.Ly // 2 + 1)


def grid_spec(
    max_a: int,
    max_b: int,
    factor: float = 1.25,
    margin: int = 16,
    min_size: int = 32,
    damping: float = 1e4,
) -> GridSpec:
    """Grid for counts up to ``(max_a, max_b)``.

    ``L = next_fast_len(max(min_size, ceil(factor * max + margin)))`` per axis (5-smooth
    sizes for a fast FFT). The radii are chosen so the round-off amplification
    ``r_x^-max_a r_y^-max_b`` at the largest needed cell is at most ``damping``
    (``1`` disables damping); a count ``a + Lx`` beyond the grid then aliases onto
    ``a`` damped by ``r_x^Lx = damping^-(Lx / (2 max_a))`` (about ``3e-3`` with the
    defaults).
    """

    def size(m):
        return scipy.fft.next_fast_len(max(int(min_size), math.ceil(factor * m + margin)))

    Lx, Ly = size(max_a), size(max_b)
    amp = max(float(damping), 1.0)
    # (a floor of 8 keeps r^-L moderate when one label is (almost) never observed)
    rx = amp ** (-1.0 / (2 * max(max_a, 8)))
    ry = amp ** (-1.0 / (2 * max(max_b, 8)))
    return GridSpec(Lx, Ly, rx, ry)


def law_and_jac(
    model: GWModel,
    P: np.ndarray,
    dP: np.ndarray,
    rows: np.ndarray,
    generations: int,
    grid: GridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Unconditional law ``p (Lx, Ly)`` and derivatives ``dp (D, Lx, Ly)``.

    ``dP`` holds, for each of ``D`` directions, the derivative of the single type row
    ``rows[d]`` it moves (``D = 0`` gives the law only). Entries of ``p`` are exact up to
    round-off amplified by at most the grid's ``damping`` factor and up to aliased mass
    damped by ``r^L`` (see :func:`grid_spec`).
    """
    Lx, Ly = grid.Lx, grid.Ly
    Ly2 = Ly // 2 + 1
    zx = grid.rx * np.exp(-2j * np.pi * np.arange(Lx) / Lx)
    zy = grid.ry * np.exp(-2j * np.pi * np.arange(Ly2) / Ly)
    D = dP.shape[0]
    out = np.empty((Lx, Ly2), np.complex128)
    dout = np.empty((D, Lx, Ly2), np.complex128)
    c0, cx, cy = model.f0
    _pgf_grid(
        np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(dP, dtype=float),
        np.asarray(rows, dtype=np.int64), model.offspring, model.n_out, int(generations),
        zx, zy, c0, cx, cy, model.founder, model.sizes, model.size_probs, out, dout,
    )  # fmt: skip
    p = scipy.fft.irfft2(out, s=(Lx, Ly), workers=1)
    dp = scipy.fft.irfft2(dout, s=(Lx, Ly), axes=(1, 2), workers=1) if D else dout.real
    scale = grid.rx ** -np.arange(Lx)[:, None] * grid.ry ** -np.arange(Ly)[None, :]
    if grid.rx != 1.0 or grid.ry != 1.0:
        p = p * scale
        if D:
            dp = dp * scale[None]
    return p, dp


def detection_and_grad(
    model: GWModel, P: np.ndarray, dP: np.ndarray, rows: np.ndarray, generations: int
) -> tuple[float, np.ndarray]:
    """``P(Z_n != 0)`` and its ``D`` directional derivatives, in complement form.

    With ``g_k = 1 - F_k(0, 0)`` (probability that a type-``k`` cell leaves a detected
    descendant), ``g_k' = sum_o P[k,o] (1 - (1-g_A)^a (1-g_B)^b)`` evaluated with
    ``expm1``/``log1p``: accurate even when detection is rare.
    """
    off = model.offspring
    g = np.asarray(model.capture, dtype=float).copy()  # F_0,k(0, 0) = 1 - c_k
    D = dP.shape[0]
    dg = np.zeros((D, 2))
    for _ in range(int(generations)):
        la = math.log1p(-g[0]) if g[0] < 1 else -np.inf
        lb = math.log1p(-g[1]) if g[1] < 1 else -np.inf
        new = np.zeros(2)
        jac = np.zeros((2, 2))  # d g_k' / d g_j
        comp = np.zeros((2, off.shape[1]))  # 1 - (1-gA)^a (1-gB)^b
        for k in range(2):
            for o in range(model.n_out[k]):
                a, b = int(off[k, o, 0]), int(off[k, o, 1])
                logm = (a * la if a else 0.0) + (b * lb if b else 0.0)
                comp[k, o] = -math.expm1(logm) if np.isfinite(logm) else 1.0
                if P[k, o] == 0:
                    continue
                new[k] += P[k, o] * comp[k, o]
                if a:
                    jac[k, 0] += P[k, o] * a * (1 - g[0]) ** (a - 1) * (1 - g[1]) ** b
                if b:
                    jac[k, 1] += P[k, o] * b * (1 - g[0]) ** a * (1 - g[1]) ** (b - 1)
        ndg = dg @ jac.T
        for d in range(D):
            r = rows[d]
            ndg[d, r] += float(dP[d, : model.n_out[r]] @ comp[r, : model.n_out[r]])
        g, dg = new, ndg
    # founder mixture: 1 - sum_m pi_m (1 - phi.g)^m
    s = float(model.founder @ g)
    ds = dg @ model.founder
    q, dq = 0.0, np.zeros(D)
    ls = math.log1p(-s) if s < 1 else -np.inf
    for m, pi in zip(model.sizes, model.size_probs, strict=True):
        if pi == 0:
            continue
        q += pi * (-math.expm1(m * ls) if np.isfinite(ls) else 1.0)
        dq += pi * m * (1 - s) ** (m - 1) * ds
    return q, dq


# ---------------------------------------------------------------------- likelihood
@dataclass
class HorizonData:
    """Count histogram of one cohort: unique ``(a, b)`` cells and their multiplicities."""

    generations: int
    a: np.ndarray
    b: np.ndarray
    weight: np.ndarray
    grid: GridSpec

    @property
    def clones(self) -> int:
        """Number of clones in the cohort."""
        return int(self.weight.sum())


def histogram(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the unique rows of an ``(n, 2)`` count matrix and their multiplicities."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2 or counts.shape[1] != 2:
        raise ValueError("Exact GW likelihood needs (n, 2) count matrices (two labels)")
    cells, weight = np.unique(counts, axis=0, return_counts=True)
    return cells[:, 0], cells[:, 1], weight.astype(float)


def log_likelihood(
    model: GWModel,
    P: np.ndarray,
    dP: np.ndarray,
    rows: np.ndarray,
    data: list[HorizonData],
    floor: float = 1e-15,
    min_detection: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Log-likelihood of all cohorts and its gradient along the ``D`` directions.

    ``sum_h sum_cells w log p_h(a, b) - [detected] N_h log max(P_h(Z != 0), min_detection)``.
    Probabilities are floored at ``floor`` (their gradient is then zero) so
    impossible-looking cells cost a finite, large penalty instead of ``-inf``.

    The detection probability is floored at ``min_detection``: the FFT gives ``p`` to an
    absolute accuracy of ~1e-15, so when detection is (almost) impossible the ratio
    ``p / P(Z != 0)`` would be numerical noise, and floored cells divided by a tiny
    detection probability create spurious, arbitrarily high likelihoods. Below the floor
    the conditional likelihood is underestimated (by ``P(Z != 0) / min_detection``) -
    only in far-away parameter regions where clones are essentially never detected.
    """
    total, grad = 0.0, np.zeros(dP.shape[0])
    for h in data:
        p, dp = law_and_jac(model, P, dP, rows, h.generations, h.grid)
        pc = p[h.a, h.b]
        ok = pc > floor
        pz = np.where(ok, pc, floor)
        total += float(h.weight @ np.log(pz))
        if dP.shape[0]:
            grad += dp[:, h.a, h.b] @ np.where(ok, h.weight / pz, 0.0)
        if model.detected:
            q, dq = detection_and_grad(model, P, dP, rows, h.generations)
            if q > min_detection:
                total -= h.clones * math.log(q)
                grad -= h.clones * dq / q
            else:
                total -= h.clones * math.log(min_detection)
    return total, grad
