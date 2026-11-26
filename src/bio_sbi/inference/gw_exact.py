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
for every direction ``d``, ``dF_n = (d_P f)[dP_d](F_{n-1}) + J_f(F_{n-1}) dF_{n-1}``
(``kernel="loop"``, the default). ``kernel="adjoint"`` (:func:`log_likelihood`) instead
writes ``f`` as a polynomial in its distinct daughter monomials and takes the gradient in
reverse mode through the inverse FFT and the recursion: one backward sweep per grid point
for all directions (same values, gradients equal up to round-off, several times faster).

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
from dataclasses import dataclass, replace

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


# ------------------------------------------------- adjoint kernel (kernel="adjoint")
# The offspring pgf of type k is a polynomial f_k(x, y) = sum_m C[k, m] x^ma[m] y^mb[m]
# over the M distinct daughter monomials of the law (M <= 6 for binary divisions). The
# primal recursion evaluates the M monomials once per step and shares them between the
# two types. Derivatives are taken in reverse mode: the log likelihood depends on the
# grid values G(i, j) only through a real-linear functional
# ``Re sum_ij A(i, j) dG(i, j)`` (adjoint of the inverse FFT, see
# :func:`_fft_adjoint`), so one backward sweep per grid point gives the gradient with
# respect to all coefficients C[k, m] at once, at a cost independent of the number of
# free parameters. Both kernels work on one grid row at a time with real and imaginary
# parts in separate arrays, so the inner loops over the row vectorise (SIMD); complex
# products are spelled out exactly as Numba's complex multiply (same results).
@njit(cache=True, nogil=True)
def _pgf_value(C, ma, mb, maxp, gens, zx, zy, F0c, F0x, F0y, phi, sizes, pis, out, traj):
    """Grid values ``G`` (no derivatives) from the polynomial form ``C (2, M)``.

    If ``traj`` has shape ``(len(zx), gens + 1, 4, len(zy))`` (else size 0), the iterates
    ``F_n`` (real A, imag A, real B, imag B) are stored for :func:`_pgf_adjoint`.
    """
    n = zy.shape[0]
    M = ma.shape[0]
    store = traj.shape[0] > 0
    yr = zy.real.copy()
    yi = zy.imag.copy()
    Ar = np.empty(n)
    Ai = np.empty(n)
    Br = np.empty(n)
    Bi = np.empty(n)
    PAr = np.empty((maxp + 1, n))
    PAi = np.empty((maxp + 1, n))
    PBr = np.empty((maxp + 1, n))
    PBi = np.empty((maxp + 1, n))
    for j in range(n):
        PAr[0, j] = 1.0
        PAi[0, j] = 0.0
        PBr[0, j] = 1.0
        PBi[0, j] = 0.0
    for i in range(zx.shape[0]):
        xr = zx[i].real
        xi = zx[i].imag
        for j in range(n):
            Ar[j] = F0c[0] + F0x[0] * xr + F0y[0] * yr[j]
            Ai[j] = F0x[0] * xi + F0y[0] * yi[j]
            Br[j] = F0c[1] + F0x[1] * xr + F0y[1] * yr[j]
            Bi[j] = F0x[1] * xi + F0y[1] * yi[j]
        for g in range(gens):
            if store:
                for j in range(n):
                    traj[i, g, 0, j] = Ar[j]
                    traj[i, g, 1, j] = Ai[j]
                    traj[i, g, 2, j] = Br[j]
                    traj[i, g, 3, j] = Bi[j]
            for e in range(1, maxp + 1):
                for j in range(n):
                    PAr[e, j] = PAr[e - 1, j] * Ar[j] - PAi[e - 1, j] * Ai[j]
                    PAi[e, j] = PAr[e - 1, j] * Ai[j] + PAi[e - 1, j] * Ar[j]
                    PBr[e, j] = PBr[e - 1, j] * Br[j] - PBi[e - 1, j] * Bi[j]
                    PBi[e, j] = PBr[e - 1, j] * Bi[j] + PBi[e - 1, j] * Br[j]
            for j in range(n):
                Ar[j] = 0.0
                Ai[j] = 0.0
                Br[j] = 0.0
                Bi[j] = 0.0
            for m in range(M):
                a = ma[m]
                b = mb[m]
                c0 = C[0, m]
                c1 = C[1, m]
                for j in range(n):
                    mr = PAr[a, j] * PBr[b, j] - PAi[a, j] * PBi[b, j]
                    mi = PAr[a, j] * PBi[b, j] + PAi[a, j] * PBr[b, j]
                    Ar[j] += c0 * mr
                    Ai[j] += c0 * mi
                    Br[j] += c1 * mr
                    Bi[j] += c1 * mi
        if store:
            for j in range(n):
                traj[i, gens, 0, j] = Ar[j]
                traj[i, gens, 1, j] = Ai[j]
                traj[i, gens, 2, j] = Br[j]
                traj[i, gens, 3, j] = Bi[j]
        for j in range(n):
            S = phi[0] * complex(Ar[j], Ai[j]) + phi[1] * complex(Br[j], Bi[j])
            G = 0.0j
            for s in range(sizes.shape[0]):
                G += pis[s] * S ** sizes[s]
            out[i, j] = G


@njit(cache=True, nogil=True)
def _pgf_adjoint(C, ma, mb, maxp, gens, phi, sizes, pis, traj, A, barC):
    """Accumulate ``barC[k, m] += Re sum_ij A(i, j) dG(i, j) / dC[k, m]`` (reverse mode).

    ``traj`` holds the iterates stored by :func:`_pgf_value`; ``A (Lx, len(zy))`` complex
    is the cotangent of the grid values; ``barC (2, M)`` real is added to (not reset).
    """
    n = A.shape[1]
    M = ma.shape[0]
    PAr = np.empty((maxp + 1, n))
    PAi = np.empty((maxp + 1, n))
    PBr = np.empty((maxp + 1, n))
    PBi = np.empty((maxp + 1, n))
    for j in range(n):
        PAr[0, j] = 1.0
        PAi[0, j] = 0.0
        PBr[0, j] = 1.0
        PBi[0, j] = 0.0
    lAr = np.empty(n)
    lAi = np.empty(n)
    lBr = np.empty(n)
    lBi = np.empty(n)
    J = np.empty((8, n))  # d f_A / dA, d f_B / dA, d f_A / dB, d f_B / dB (re, im)
    acc = np.zeros((2, M, n))
    for i in range(A.shape[0]):
        for j in range(n):
            S = phi[0] * complex(traj[i, gens, 0, j], traj[i, gens, 1, j]) + phi[1] * complex(
                traj[i, gens, 2, j], traj[i, gens, 3, j]
            )
            dG = 0.0j
            for s in range(sizes.shape[0]):
                k = sizes[s]
                if k > 0:
                    dG += pis[s] * k * S ** (k - 1)
            lam = A[i, j] * dG
            lA = lam * phi[0]
            lB = lam * phi[1]
            lAr[j] = lA.real
            lAi[j] = lA.imag
            lBr[j] = lB.real
            lBi[j] = lB.imag
        for g in range(gens - 1, -1, -1):
            # step g maps F_g -> F_{g+1}; (lA, lB) is the cotangent of F_{g+1}
            for e in range(1, maxp + 1):
                for j in range(n):
                    Ar = traj[i, g, 0, j]
                    Ai = traj[i, g, 1, j]
                    Br = traj[i, g, 2, j]
                    Bi = traj[i, g, 3, j]
                    PAr[e, j] = PAr[e - 1, j] * Ar - PAi[e - 1, j] * Ai
                    PAi[e, j] = PAr[e - 1, j] * Ai + PAi[e - 1, j] * Ar
                    PBr[e, j] = PBr[e - 1, j] * Br - PBi[e - 1, j] * Bi
                    PBi[e, j] = PBr[e - 1, j] * Bi + PBi[e - 1, j] * Br
            for m in range(M):
                a = ma[m]
                b = mb[m]
                for j in range(n):
                    mr = PAr[a, j] * PBr[b, j] - PAi[a, j] * PBi[b, j]
                    mi = PAr[a, j] * PBi[b, j] + PAi[a, j] * PBr[b, j]
                    acc[0, m, j] += lAr[j] * mr - lAi[j] * mi
                    acc[1, m, j] += lBr[j] * mr - lBi[j] * mi
            if g == 0:
                break
            J[:] = 0.0
            for m in range(M):
                a = ma[m]
                b = mb[m]
                c0 = C[0, m]
                c1 = C[1, m]
                if a > 0:
                    for j in range(n):
                        gr = a * (PAr[a - 1, j] * PBr[b, j] - PAi[a - 1, j] * PBi[b, j])
                        gi = a * (PAr[a - 1, j] * PBi[b, j] + PAi[a - 1, j] * PBr[b, j])
                        J[0, j] += c0 * gr
                        J[1, j] += c0 * gi
                        J[2, j] += c1 * gr
                        J[3, j] += c1 * gi
                if b > 0:
                    for j in range(n):
                        gr = b * (PAr[a, j] * PBr[b - 1, j] - PAi[a, j] * PBi[b - 1, j])
                        gi = b * (PAr[a, j] * PBi[b - 1, j] + PAi[a, j] * PBr[b - 1, j])
                        J[4, j] += c0 * gr
                        J[5, j] += c0 * gi
                        J[6, j] += c1 * gr
                        J[7, j] += c1 * gi
            for j in range(n):
                # cotangent of F_g: (lA, lB) J, J[k, l] = d f_k / d F_l
                nAr = lAr[j] * J[0, j] - lAi[j] * J[1, j] + lBr[j] * J[2, j] - lBi[j] * J[3, j]
                nAi = lAr[j] * J[1, j] + lAi[j] * J[0, j] + lBr[j] * J[3, j] + lBi[j] * J[2, j]
                nBr = lAr[j] * J[4, j] - lAi[j] * J[5, j] + lBr[j] * J[6, j] - lBi[j] * J[7, j]
                nBi = lAr[j] * J[5, j] + lAi[j] * J[4, j] + lBr[j] * J[7, j] + lBi[j] * J[6, j]
                lAr[j] = nAr
                lAi[j] = nAi
                lBr[j] = nBr
                lBi[j] = nBi
    for k in range(2):
        for m in range(M):
            t = 0.0
            for j in range(n):
                t += acc[k, m, j]
            barC[k, m] += t


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


def monomials(model: GWModel) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Distinct daughter monomials ``x^ma y^mb`` of the law and each outcome's index.

    Returns ``(ma, mb, index, maxp)``: ``index (2, O)`` maps outcome ``o`` of type ``k``
    to its monomial (``-1`` for padding outcomes), ``maxp`` is the largest exponent.
    """
    off = np.asarray(model.offspring)
    pairs: list[tuple[int, int]] = []
    index = np.full(off.shape[:2], -1, dtype=np.int64)
    for k in range(2):
        for o in range(int(model.n_out[k])):
            ab = (int(off[k, o, 0]), int(off[k, o, 1]))
            if ab not in pairs:
                pairs.append(ab)
            index[k, o] = pairs.index(ab)
    ma = np.array([a for a, _ in pairs], dtype=np.int64)
    mb = np.array([b for _, b in pairs], dtype=np.int64)
    return ma, mb, index, int(max(ma.max(), mb.max(), 1))


def poly_coefficients(
    index: np.ndarray, n_monomials: int, P: np.ndarray, dP: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial form ``C (2, M)`` of ``P`` and ``dC (D, M)`` of the direction rows."""
    C = np.zeros((2, n_monomials))
    for k in range(2):
        ok = index[k] >= 0
        np.add.at(C[k], index[k][ok], np.asarray(P, dtype=float)[k][ok])
    dC = np.zeros((dP.shape[0], n_monomials))
    for d in range(dP.shape[0]):
        idx = index[rows[d]]
        ok = idx >= 0
        np.add.at(dC[d], idx[ok], dP[d][ok])
    return C, dC


def _grid_points(grid: GridSpec) -> tuple[np.ndarray, np.ndarray]:
    zx = grid.rx * np.exp(-2j * np.pi * np.arange(grid.Lx) / grid.Lx)
    zy = grid.ry * np.exp(-2j * np.pi * np.arange(grid.Ly // 2 + 1) / grid.Ly)
    return zx, zy


def _scale(grid: GridSpec) -> np.ndarray:
    return grid.rx ** -np.arange(grid.Lx)[:, None] * grid.ry ** -np.arange(grid.Ly)[None, :]


def law_poly(model: GWModel, C: np.ndarray, mono: tuple, generations: int, grid: GridSpec,
             store: bool = False) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Unconditional law ``p (Lx, Ly)`` from the polynomial form (adjoint kernel).

    Returns ``(p, traj)``; ``traj`` holds the pgf iterates for :func:`_pgf_adjoint` if
    ``store`` (else it is empty).
    """
    ma, mb, _, maxp = mono
    gens = int(generations)
    zx, zy = _grid_points(grid)
    out = np.empty((grid.Lx, len(zy)), np.complex128)
    shape = (grid.Lx, gens + 1, 4, len(zy)) if store else (0, 0, 0, 0)
    traj = np.empty(shape)
    c0, cx, cy = model.f0
    _pgf_value(C, ma, mb, maxp, gens, zx, zy, c0, cx, cy, model.founder, model.sizes,
               model.size_probs, out, traj)  # fmt: skip
    p = scipy.fft.irfft2(out, s=(grid.Lx, grid.Ly), workers=1)
    if grid.rx != 1.0 or grid.ry != 1.0:
        p = p * _scale(grid)
    return p, traj


def _fft_adjoint(cot: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Cotangent ``A`` of the half-grid values for a cotangent ``cot (Lx, Ly)`` of ``p``.

    ``sum_ab cot[a, b] irfft2(D)[a, b] = Re sum_ij A[i, j] D[i, j]`` for any half-grid
    array ``D``: ``irfft2`` is ``(1/(Lx Ly)) Re sum_ij w_j D[i, j] e^{+2 pi i (...)}`` with
    ``w_j = 2`` except for the zero (and, for even ``Ly``, Nyquist) column.
    """
    Lx, Ly = grid.Lx, grid.Ly
    A = np.conj(scipy.fft.rfft2(cot, workers=1)) / (Lx * Ly)
    A[:, 1:] *= 2.0
    if Ly % 2 == 0:
        A[:, -1] *= 0.5
    return A


@njit(cache=True, nogil=True)
def _detection_nb(off, n_out, capture, P, dP, rows, gens, founder, sizes, size_probs):
    """Numba port of :func:`detection_and_grad` (same formulas; adjoint kernel)."""
    D = dP.shape[0]
    n_cols = off.shape[1]
    g = capture.copy()
    dg = np.zeros((D, 2))
    comp = np.zeros((2, n_cols))
    for _ in range(gens):
        la = math.log1p(-g[0]) if g[0] < 1 else -np.inf
        lb = math.log1p(-g[1]) if g[1] < 1 else -np.inf
        new = np.zeros(2)
        jac = np.zeros((2, 2))
        for k in range(2):
            for o in range(n_out[k]):
                a = off[k, o, 0]
                b = off[k, o, 1]
                logm = (a * la if a else 0.0) + (b * lb if b else 0.0)
                comp[k, o] = -math.expm1(logm) if np.isfinite(logm) else 1.0
                if P[k, o] == 0:
                    continue
                new[k] += P[k, o] * comp[k, o]
                if a:
                    jac[k, 0] += P[k, o] * a * (1 - g[0]) ** (a - 1) * (1 - g[1]) ** b
                if b:
                    jac[k, 1] += P[k, o] * b * (1 - g[0]) ** a * (1 - g[1]) ** (b - 1)
        ndg = np.zeros((D, 2))
        for d in range(D):
            for k in range(2):
                ndg[d, k] = dg[d, 0] * jac[k, 0] + dg[d, 1] * jac[k, 1]
            r = rows[d]
            t = 0.0
            for o in range(n_out[r]):
                t += dP[d, o] * comp[r, o]
            ndg[d, r] += t
        g = new
        dg = ndg
    s = founder[0] * g[0] + founder[1] * g[1]
    ls = math.log1p(-s) if s < 1 else -np.inf
    q = 0.0
    dq = np.zeros(D)
    for i in range(sizes.shape[0]):
        pi = size_probs[i]
        if pi == 0:
            continue
        m = sizes[i]
        q += pi * (-math.expm1(m * ls) if np.isfinite(ls) else 1.0)
        f = pi * m * (1 - s) ** (m - 1)
        for d in range(D):
            dq[d] += f * (dg[d, 0] * founder[0] + dg[d, 1] * founder[1])
    return q, dq


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
        """Number of clones in the cohort (rounded: minibatch weights are rescaled)."""
        return int(round(float(self.weight.sum())))


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
    kernel: str = "loop",
) -> tuple[float, np.ndarray]:
    """Log-likelihood of all cohorts and its gradient along the ``D`` directions.

    ``kernel``: ``"loop"`` (default) propagates the ``D`` directions in forward mode
    through the pgf recursion (:func:`law_and_jac`); ``"adjoint"`` evaluates the pgf in
    polynomial form and takes the gradient in reverse mode (one backward sweep per grid
    point for all directions, adjoint of the inverse FFT). Both give the same values and
    gradients up to round-off; ``"adjoint"`` is faster, increasingly so with ``D``.

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
    if kernel not in ("loop", "adjoint"):
        raise ValueError(f"Unknown exact-GW kernel {kernel!r} (loop | adjoint)")
    total, grad = 0.0, np.zeros(dP.shape[0])
    if kernel == "adjoint":
        mono = monomials(model)
        C, dC = poly_coefficients(mono[2], len(mono[0]), P, dP, rows)
        barC = np.zeros((2, len(mono[0])))
    for h in data:
        if kernel == "adjoint":
            p, traj = law_poly(model, C, mono, h.generations, h.grid, store=dP.shape[0] > 0)
        else:
            p, dp = law_and_jac(model, P, dP, rows, h.generations, h.grid)
        pc = p[h.a, h.b]
        ok = pc > floor
        pz = np.where(ok, pc, floor)
        total += float(h.weight @ np.log(pz))
        if dP.shape[0] and kernel == "adjoint":
            g = h.grid
            # d log-lik / d p on the observed cells, times d p / d (FFT output) = scale
            cot = np.zeros((g.Lx, g.Ly))
            cot[h.a, h.b] = np.where(ok, h.weight / pz, 0.0) * (
                g.rx ** -h.a.astype(float) * g.ry ** -h.b.astype(float)
            )  # histogram cells are unique
            A = _fft_adjoint(cot, g)
            _pgf_adjoint(C, mono[0], mono[1], mono[3], int(h.generations), model.founder,
                         model.sizes, model.size_probs, traj, A, barC)  # fmt: skip
        elif dP.shape[0]:
            grad += dp[:, h.a, h.b] @ np.where(ok, h.weight / pz, 0.0)
        if model.detected and kernel == "adjoint":
            q, dq = _detection_nb(model.offspring, model.n_out, np.asarray(model.capture, float),
                                  np.ascontiguousarray(P, dtype=float),
                                  np.ascontiguousarray(dP, dtype=float),
                                  np.asarray(rows, dtype=np.int64), int(h.generations),
                                  model.founder, model.sizes, model.size_probs)  # fmt: skip
        elif model.detected:
            q, dq = detection_and_grad(model, P, dP, rows, h.generations)
        if model.detected:
            if q > min_detection:
                total -= h.clones * math.log(q)
                grad -= h.clones * dq / q
            else:
                total -= h.clones * math.log(min_detection)
    if kernel == "adjoint" and dP.shape[0]:
        grad += np.einsum("dm,dm->d", dC, barC[np.asarray(rows, dtype=np.int64)])
    return total, grad


# ------------------------------------------------- free founder distribution
# With the founder-type probabilities ``phi`` and the founder-size probabilities ``pi``
# inferred, the grid values are G = sum_m pi_m S^m with S = phi_A F_A + phi_B F_B, where
# F_k is the pgf of a clone started by one type-k cell. Hence
#   dG/dphi_k = (sum_m pi_m m S^(m-1)) F_k,   dG/dpi_m = S^m,
# and the same for the detection probability q = sum_m pi_m (1 - (1 - phi.g)^m) with
# g_k = P(a type-k cell leaves a detected descendant):
#   dq/dphi_k = (sum_m pi_m m (1 - s)^(m-1)) g_k,   dq/dpi_m = 1 - (1 - s)^m.
_ONE = np.array([1], dtype=np.int64)
_UNIT = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))


def single_founder_pgfs(model: GWModel, C: np.ndarray, mono: tuple, generations: int,
                        grid: GridSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Pgfs ``F_A, F_B`` of a clone started by one A / one B cell on the half grid.

    Returns ``(FA, FB, traj)`` (complex ``(Lx, Ly // 2 + 1)`` each; ``traj`` = the stored
    iterates for :func:`_pgf_adjoint`). Polynomial form ``C`` as for :func:`law_poly`.
    """
    ma, mb, _, maxp = mono
    gens = int(generations)
    zx, zy = _grid_points(grid)
    out = np.empty((grid.Lx, len(zy)), np.complex128)
    traj = np.empty((grid.Lx, gens + 1, 4, len(zy)))
    c0, cx, cy = model.f0
    _pgf_value(C, ma, mb, maxp, gens, zx, zy, c0, cx, cy, _UNIT[0], _ONE, np.ones(1), out, traj)
    FA = traj[:, gens, 0] + 1j * traj[:, gens, 1]
    FB = traj[:, gens, 2] + 1j * traj[:, gens, 3]
    return FA, FB, traj


def _loop_single_founder(model: GWModel, P, dP, rows, generations: int, grid: GridSpec):
    """``F_k`` and ``dF_k (D, ...)`` on the half grid for k = A, B (forward mode)."""
    zx, zy = _grid_points(grid)
    c0, cx, cy = model.f0
    res = []
    for e in _UNIT:
        out = np.empty((grid.Lx, len(zy)), np.complex128)
        dout = np.empty((dP.shape[0], grid.Lx, len(zy)), np.complex128)
        _pgf_grid(
            np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(dP, dtype=float),
            np.asarray(rows, dtype=np.int64), model.offspring, model.n_out, int(generations),
            zx, zy, c0, cx, cy, e, _ONE, np.ones(1), out, dout,
        )  # fmt: skip
        res.extend([out, dout])
    return res


def _mix(phi, sizes, pis, FA, FB):
    """``G``, ``dG/dS``, ``dG/dphi (2, ...)`` and ``dG/dpi (S, ...)`` on the grid."""
    S = phi[0] * FA + phi[1] * FB
    G = np.zeros_like(S)
    dGdS = np.zeros_like(S)
    dpi = []
    for m, pi in zip(sizes, pis, strict=True):
        Sm = S ** int(m)
        G += pi * Sm
        dGdS += pi * int(m) * S ** (int(m) - 1)
        dpi.append(Sm)
    return G, dGdS, np.stack([dGdS * FA, dGdS * FB]), np.stack(dpi)


def _irfft(vals: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Coefficients (rescaled by the damping radii) of half-grid values ``(..., Lx, Ly2)``."""
    p = scipy.fft.irfft2(vals, s=(grid.Lx, grid.Ly), axes=(-2, -1), workers=1)
    if grid.rx != 1.0 or grid.ry != 1.0:
        p = p * _scale(grid)
    return p


def _founder_detection(model: GWModel, P, dP, rows, generations: int, kernel: str):
    """Return the detection probability and its gradients along ``dP``, ``phi`` and ``pi``."""
    gs, dgs = [], []
    for e in _UNIT:
        if kernel == "adjoint":
            g, dg = _detection_nb(model.offspring, model.n_out,
                                  np.asarray(model.capture, float),
                                  np.ascontiguousarray(P, dtype=float),
                                  np.ascontiguousarray(dP, dtype=float),
                                  np.asarray(rows, dtype=np.int64), int(generations), e, _ONE,
                                  np.ones(1))  # fmt: skip
        else:
            g, dg = detection_and_grad(replace(model, founder=e, sizes=_ONE,
                                               size_probs=np.ones(1)), P, dP, rows, generations)  # fmt: skip
        gs.append(float(g))
        dgs.append(np.asarray(dg, dtype=float))
    phi = np.asarray(model.founder, dtype=float)
    s = phi[0] * gs[0] + phi[1] * gs[1]
    ds = phi[0] * dgs[0] + phi[1] * dgs[1]
    ls = math.log1p(-s) if s < 1 else -np.inf
    q, dqds = 0.0, 0.0
    dq_pi = np.zeros(len(model.sizes))
    for i, (m, pi) in enumerate(zip(model.sizes, model.size_probs, strict=True)):
        c = -math.expm1(int(m) * ls) if np.isfinite(ls) else 1.0
        dq_pi[i] = c
        q += pi * c
        dqds += pi * int(m) * (1 - s) ** (int(m) - 1)
    return q, dqds * ds, dqds * np.asarray(gs), dq_pi


def log_likelihood_founder(
    model: GWModel,
    P: np.ndarray,
    dP: np.ndarray,
    rows: np.ndarray,
    data: list[HorizonData],
    floor: float = 1e-15,
    min_detection: float = 1e-6,
    kernel: str = "loop",
    grad: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """:func:`log_likelihood` plus the gradient in the founder distribution.

    Returns ``(log-likelihood, d/d directions (D,), d/d model.founder (2,),
    d/d model.size_probs (S,))``, the last two treating every entry as a free variable
    (the caller applies the simplex constraint). Same value as :func:`log_likelihood` up
    to round-off; ``grad=False`` returns zero gradients. Both kernels: ``adjoint`` takes
    the founder gradients by contracting the FFT adjoint with ``dG/dphi``, ``dG/dpi``
    (cheap: final-generation quantities only); ``loop`` propagates ``F_A`` and ``F_B``
    separately in forward mode (twice the cost of :func:`log_likelihood`).
    """
    if kernel not in ("loop", "adjoint"):
        raise ValueError(f"Unknown exact-GW kernel {kernel!r} (loop | adjoint)")
    D = dP.shape[0] if grad else 0
    if not grad:
        dP, rows = dP[:0], rows[:0]
    phi = np.asarray(model.founder, dtype=float)
    sizes, pis = np.asarray(model.sizes), np.asarray(model.size_probs, dtype=float)
    total, g_dir = 0.0, np.zeros(D)
    g_phi, g_pi = np.zeros(2), np.zeros(len(sizes))
    if kernel == "adjoint":
        mono = monomials(model)
        C, dC = poly_coefficients(mono[2], len(mono[0]), P, dP, rows)
        barC = np.zeros((2, len(mono[0])))
    for h in data:
        g = h.grid
        if kernel == "adjoint":
            FA, FB, traj = single_founder_pgfs(model, C, mono, h.generations, g)
            G, dGdS, dG_phi, dG_pi = _mix(phi, sizes, pis, FA, FB)
            p = _irfft(G, g)
        else:
            FA, dFA, FB, dFB = _loop_single_founder(model, P, dP, rows, h.generations, g)
            G, dGdS, dG_phi, dG_pi = _mix(phi, sizes, pis, FA, FB)
            p = _irfft(G, g)
        pc = p[h.a, h.b]
        ok = pc > floor
        pz = np.where(ok, pc, floor)
        total += float(h.weight @ np.log(pz))
        if grad:
            wv = np.where(ok, h.weight / pz, 0.0)
            if kernel == "adjoint":
                cot = np.zeros((g.Lx, g.Ly))
                cot[h.a, h.b] = wv * (g.rx ** -h.a.astype(float) * g.ry ** -h.b.astype(float))
                A = _fft_adjoint(cot, g)
                g_phi += np.real(np.einsum("ij,kij->k", A, dG_phi))
                g_pi += np.real(np.einsum("ij,kij->k", A, dG_pi))
                if D:
                    # the adjoint kernel's chain rule from S to F_A, F_B is lam * phi_k
                    _pgf_adjoint(C, mono[0], mono[1], mono[3], int(h.generations), phi, sizes,
                                 pis, traj, A, barC)  # fmt: skip
            else:
                extra = _irfft(np.concatenate([dG_phi, dG_pi]), g)
                ge = extra[:, h.a, h.b] @ wv
                g_phi += ge[:2]
                g_pi += ge[2:]
                if D:
                    dG = dGdS[None] * (phi[0] * dFA + phi[1] * dFB)
                    g_dir += _irfft(dG, g)[:, h.a, h.b] @ wv
        if model.detected:
            q, dq, dq_phi, dq_pi = _founder_detection(model, P, dP, rows, h.generations, kernel)
            if q > min_detection:
                total -= h.clones * math.log(q)
                if grad:
                    g_dir -= h.clones * dq / q
                    g_phi -= h.clones * dq_phi / q
                    g_pi -= h.clones * dq_pi / q
            else:
                total -= h.clones * math.log(min_detection)
    if kernel == "adjoint" and D:
        g_dir += np.einsum("dm,dm->d", dC, barC[np.asarray(rows, dtype=np.int64)])
    return total, g_dir, g_phi, g_pi


def founder_composition_laws(model: GWModel, P: np.ndarray, generations: int,
                             grid: GridSpec) -> tuple[list[str], np.ndarray]:  # fmt: skip
    """Joint law of the counts and the founder composition (unconditional on detection).

    Returns ``(labels, p)`` with ``p (K, Lx, Ly)``: ``p[c, a, b] = P(founders = c,
    Z = (a, b))`` for every composition ``c`` = (founder count ``m``, ``k`` A founders)
    with ``pi_m > 0``, labelled ``"A"``, ``"B"`` (``m = 1``), ``"AA"``, ``"AB"``, ``"BB"``
    (``m = 2``), ... Summing over ``c`` gives the law of :func:`law_poly`. Posterior
    composition probabilities of a clone with counts ``(a, b) != (0, 0)`` are
    ``p[:, a, b] / p[:, a, b].sum()`` (detection conditioning cancels).
    """
    mono = monomials(model)
    C, _ = poly_coefficients(mono[2], len(mono[0]), P, np.zeros((0, P.shape[1])),
                            np.zeros(0, dtype=np.int64))  # fmt: skip
    FA, FB, _ = single_founder_pgfs(model, C, mono, generations, grid)
    phi = np.asarray(model.founder, dtype=float)
    labels, vals = [], []
    for m, pi in zip(model.sizes, model.size_probs, strict=True):
        m = int(m)
        if pi <= 0:
            continue
        for k in range(m, -1, -1):  # k A founders
            w = pi * math.comb(m, k) * phi[0] ** k * phi[1] ** (m - k)
            if w <= 0:
                continue
            labels.append("A" * k + "B" * (m - k))
            vals.append(w * FA**k * FB ** (m - k))
    return labels, np.clip(_irfft(np.stack(vals), grid), 0.0, None)
