"""Exact endpoint law of a two-type Bellman-Harris clone (integer Gamma shapes).

Model (:mod:`bio_sbi.simulators.bellman_harris`, ``bh_forward``): every cell of type
``k`` lives ``Gamma(m_k, mean_k / m_k)``; at the end of its life it draws one outcome
``o`` from row ``k`` and is replaced by that outcome's daughters (``offspring[k, o]``
cells of each type, taken from :class:`~bio_sbi.simulators.law.OffspringLaw`, so outcome
names and their order never matter), all born at that instant with age 0. Founders are
born at ``t = 0``. The data are the cells alive at the horizon.

Method of stages. ``Gamma(m, mean / m)`` is the sum of ``m`` exponential phases of rate
``lambda_k = m_k / mean_k``. With ``G_{k,j}(x, y, t)`` the pgf of the population at time
``t`` of a clone started by one type-``k`` cell at the start of phase ``j``, the
backward Kolmogorov equations are (one independent ODE system per grid point ``(x, y)``)

    dG_{k,j}/dt = lambda_k (G_{k,j+1} - G_{k,j}),                   j = 1 .. m_k - 1
    dG_{k,m}/dt = lambda_k (f_k(G_{A,1}, G_{B,1}) - G_{k,m}),
    G_{k,j}(0)  = F0_k(x, y)   (x for A, y for B with a perfect assay),

with ``f_k(x, y) = sum_o P[k, o] x^{a_ko} y^{b_ko}`` the offspring pgf. The clone pgf
is ``G(x, y) = sum_n pi_n (phi_A G_{A,1} + phi_B G_{B,1})^n`` (founder types ``phi``,
founder sizes ``n``). Observation enters through the initial condition, exactly as in
:mod:`~bio_sbi.inference.gw_exact`: ``F0_k = 1 - c_k + c_k (C[k,0] x + C[k,1] y)``.

The systems are integrated with a **fixed-step explicit Runge-Kutta method** (DOP853
tableau, 8th order; default ``dt = 1 h``; halving ``dt`` changes the law by < 1e-15,
doubling it by ~2e-13, see ``scripts/b_row_theory/bhexact_validation.md``) in a Numba
kernel that releases the GIL and works on a range of grid points, so one evaluation is
split over threads. One trajectory serves every horizon (the solution is recorded at
each). The grid and the FFT follow :func:`~bio_sbi.inference.gw_exact.law_and_jac`:
``Lx x Ly`` points ``z = r w`` (roots of unity ``w``, damping radius ``r <= 1``), only
half of the B axis (the law is real), ``p = irfft2(G) * r^-(a + b)``.

Forward-mode tangents give exact derivatives of the discrete solution along ``D``
directions: a direction either moves one outcome row by ``dP[d]`` (``kind 0``) or is
``d / d log mean_k`` (``kind 1``, ``d lambda_k = -lambda_k``).

The detection probability ``1 - G(0, 0)`` is integrated separately in complement form
(``g = 1 - G`` with ``expm1`` / ``log1p``), so it is accurate even when detection is rare.

Non-integer shapes have no finite phase representation (see :func:`check_shapes`).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.fft
from numba import njit

from .gw_exact import GridSpec

#: Default fixed step (hours) and Runge-Kutta tableau.
DEFAULT_DT = 1.0
DEFAULT_INTEGRATOR = "dop853"
#: Largest supported Gamma shape (cost is linear in the number of phases).
MAX_SHAPE = 256


# ---------------------------------------------------------------------- tableaus
def tableau(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Explicit Runge-Kutta tableau ``(A, b)``: ``"dop853"`` (8th order) or ``"rk4"``."""
    if name == "rk4":
        A = np.zeros((4, 4))
        A[1, 0] = A[2, 1] = 0.5
        A[3, 2] = 1.0
        return A, np.array([1.0, 2.0, 2.0, 1.0]) / 6
    if name == "dop853":
        from scipy.integrate._ivp import dop853_coefficients as c

        n = c.N_STAGES
        return np.ascontiguousarray(c.A[:n, :n], dtype=float), np.ascontiguousarray(c.B, float)
    raise ValueError(f"Unknown integrator {name!r} (dop853 | rk4)")


def segments(horizons, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Return steps and step size between consecutive horizons (positive, increasing)."""
    hs = np.asarray(horizons, dtype=float)
    if hs.ndim != 1 or len(hs) == 0 or hs[0] <= 0 or np.any(np.diff(hs) <= 0):
        raise ValueError("horizons must be positive and strictly increasing")
    if not dt > 0:
        raise ValueError("dt must be positive")
    lens = np.diff(np.r_[0.0, hs])
    n = np.maximum(1, np.ceil(lens / dt - 1e-9)).astype(np.int64)
    return n, lens / n


def check_shapes(shapes) -> np.ndarray:
    """Phases per type; ``NotImplementedError`` for non-integer Gamma shapes."""
    s = np.asarray(shapes, dtype=float)
    if np.any(np.abs(s - np.round(s)) > 1e-9) or np.any(s < 1):
        raise NotImplementedError(
            f"Exact Bellman-Harris likelihood: lifetime_shape {s.tolist()} is not a positive "
            "integer. The method of stages needs integer Gamma shapes; general lifetimes "
            "need the Bellman-Harris integral equation F_k(t) = F0_k (1 - L_k(t)) + "
            "int_0^t f_k(F(t - u)) dL_k(u) (L_k the lifetime cdf), solved on the grid by "
            "product integration / convolution in t (not implemented)."
        )
    m = np.round(s).astype(np.int64)
    if np.any(m > MAX_SHAPE):
        raise NotImplementedError(f"lifetime_shape above {MAX_SHAPE} is not supported")
    return m


# ---------------------------------------------------------------------- numba kernels
# Grid points are processed in blocks of BLOCK: every state array is (..., BLOCK) so the
# inner loops run over contiguous points and vectorise (SIMD), ~4-5x faster than one
# point at a time. Points of a partial last block are padded with zeros (results unused).
BLOCK = 32


@njit(cache=True, nogil=True, fastmath=True)
def _max_daughters(off, n_out):
    maxp = 0
    for k in range(2):
        for o in range(n_out[k]):
            for j in range(2):
                maxp = max(maxp, off[k, o, j])
    return maxp


@njit(cache=True, nogil=True, fastmath=True)
def _init_block(G, blk, nb, zx, zy, f0c, f0x, f0y, mA, mB):
    ny = zy.shape[0]
    NB = G.shape[1]
    for p in range(NB):
        if p < nb:
            i = blk + p
            x = zx[i // ny]
            y = zy[i % ny]
            gA = f0c[0] + f0x[0] * x + f0y[0] * y
            gB = f0c[1] + f0x[1] * x + f0y[1] * y
        else:
            gA = 0.0j
            gB = 0.0j
        for j in range(mA):
            G[j, p] = gA
        for j in range(mB):
            G[mA + j, p] = gB


@njit(cache=True, nogil=True, fastmath=True)
def _axpy(y, c, x):
    """``y += c x`` for C-contiguous arrays of equal shape (no temporaries)."""
    yv = y.reshape(-1)
    xv = x.reshape(-1)
    for i in range(yv.shape[0]):
        yv[i] += c * xv[i]


@njit(cache=True, nogil=True, fastmath=True)
def _stage_input(T, G, K, A, st, h):
    """``T = G + h sum_{r < st} A[st, r] K[r]`` (C-contiguous, any trailing shape)."""
    tv = T.reshape(-1)
    gv = G.reshape(-1)
    for i in range(tv.shape[0]):
        tv[i] = gv[i]
    for r in range(st):
        a = A[st, r]
        if a != 0.0:
            _axpy(T, h * a, K[r])


@njit(cache=True, nogil=True, fastmath=True)
def _powers(T, mA, maxp, powA, powB):
    NB = T.shape[1]
    for p in range(NB):
        powA[0, p] = 1.0
        powB[0, p] = 1.0
    for e in range(1, maxp + 1):
        for p in range(NB):
            powA[e, p] = powA[e - 1, p] * T[0, p]
            powB[e, p] = powB[e - 1, p] * T[mA, p]


@njit(cache=True, nogil=True, fastmath=True)
def _chain(T, f, lam, mA, mB, out):
    """Phase-chain right-hand side given the offspring pgfs ``f (2, NB)``."""
    NB = T.shape[1]
    M = mA + mB
    l0 = lam[0]
    l1 = lam[1]
    for j in range(mA - 1):
        for p in range(NB):
            out[j, p] = l0 * (T[j + 1, p] - T[j, p])
    for p in range(NB):
        out[mA - 1, p] = l0 * (f[0, p] - T[mA - 1, p])
    for j in range(mB - 1):
        for p in range(NB):
            out[mA + j, p] = l1 * (T[mA + j + 1, p] - T[mA + j, p])
    for p in range(NB):
        out[M - 1, p] = l1 * (f[1, p] - T[M - 1, p])


@njit(cache=True, nogil=True, fastmath=True)
def _offspring(P, off, n_out, powA, powB, f):
    NB = f.shape[1]
    for k in range(2):
        for p in range(NB):
            f[k, p] = 0.0
        for o in range(n_out[k]):
            pr = P[k, o]
            if pr != 0.0:
                a = off[k, o, 0]
                b = off[k, o, 1]
                for p in range(NB):
                    f[k, p] += pr * powA[a, p] * powB[b, p]


@njit(cache=True, nogil=True, fastmath=True)
def _founder_mix(G, mA, phi, sizes, pis, p):
    S = phi[0] * G[0, p] + phi[1] * G[mA, p]
    val = 0.0j
    dval = 0.0j
    for q in range(sizes.shape[0]):
        n = sizes[q]
        val += pis[q] * S**n
        dval += pis[q] * n * S ** (n - 1)
    return val, dval


@njit(cache=True, nogil=True, fastmath=True)
def _law_points(  # noqa: D417 (grouped argument descriptions)
    i0, i1, zx, zy, P, off, n_out, mA, mB, lam, f0c, f0x, f0y, phi, sizes, pis,
    seg_n, seg_dt, A, b, out,
):  # fmt: skip
    """Clone pgf at grid points ``i0 <= i < i1`` (``i = ix * len(zy) + iy``) per horizon.

    ``out`` is ``(H, len(zx) * len(zy))`` complex; row ``h`` holds the pgf at horizon ``h``.
    """
    M = mA + mB
    ns = b.shape[0]
    NB = BLOCK
    maxp = _max_daughters(off, n_out)
    G = np.empty((M, NB), np.complex128)
    T = np.empty((M, NB), np.complex128)
    K = np.empty((ns, M, NB), np.complex128)
    powA = np.empty((maxp + 1, NB), np.complex128)
    powB = np.empty((maxp + 1, NB), np.complex128)
    f = np.empty((2, NB), np.complex128)
    for blk in range(i0, i1, NB):
        nb = min(NB, i1 - blk)
        _init_block(G, blk, nb, zx, zy, f0c, f0x, f0y, mA, mB)
        for s in range(seg_n.shape[0]):
            h = seg_dt[s]
            for _ in range(seg_n[s]):
                for st in range(ns):
                    _stage_input(T, G, K, A, st, h)
                    _powers(T, mA, maxp, powA, powB)
                    _offspring(P, off, n_out, powA, powB, f)
                    _chain(T, f, lam, mA, mB, K[st])
                for r in range(ns):
                    _axpy(G, h * b[r], K[r])
            for p in range(nb):
                out[s, blk + p] = _founder_mix(G, mA, phi, sizes, pis, p)[0]


@njit(cache=True, nogil=True, fastmath=True)
def _law_points_tan(  # noqa: D417
    i0, i1, zx, zy, P, dP, kind, rows, off, n_out, mA, mB, lam, f0c, f0x, f0y, phi, sizes,
    pis, seg_n, seg_dt, A, b, out, dout,
):  # fmt: skip
    """As :func:`_law_points` plus ``dout (H, D, npts)``: derivatives along ``D`` directions.

    Direction ``d`` moves row ``rows[d]`` by ``dP[d]`` (``kind[d] == 0``) or is
    ``d / d log mean_{rows[d]}`` (``kind[d] == 1``).
    """
    M = mA + mB
    ns = b.shape[0]
    nd = dP.shape[0]
    NB = BLOCK
    maxp = _max_daughters(off, n_out)
    G = np.empty((M, NB), np.complex128)
    T = np.empty((M, NB), np.complex128)
    K = np.empty((ns, M, NB), np.complex128)
    Dg = np.empty((nd, M, NB), np.complex128)
    DT = np.empty((nd, M, NB), np.complex128)
    DK = np.empty((ns, nd, M, NB), np.complex128)
    powA = np.empty((maxp + 1, NB), np.complex128)
    powB = np.empty((maxp + 1, NB), np.complex128)
    f = np.empty((2, NB), np.complex128)
    fx = np.empty((2, NB), np.complex128)
    fy = np.empty((2, NB), np.complex128)
    df = np.empty((2, NB), np.complex128)
    start = np.array([0, mA])
    last = np.array([mA - 1, M - 1])
    for blk in range(i0, i1, NB):
        nb = min(NB, i1 - blk)
        _init_block(G, blk, nb, zx, zy, f0c, f0x, f0y, mA, mB)
        Dg[:] = 0.0
        for s in range(seg_n.shape[0]):
            h = seg_dt[s]
            for _ in range(seg_n[s]):
                for st in range(ns):
                    _stage_input(T, G, K, A, st, h)
                    _stage_input(DT, Dg, DK, A, st, h)
                    _powers(T, mA, maxp, powA, powB)
                    _offspring(P, off, n_out, powA, powB, f)
                    # Jacobian of the offspring pgfs
                    for k in range(2):
                        for p in range(NB):
                            fx[k, p] = 0.0
                            fy[k, p] = 0.0
                        for o in range(n_out[k]):
                            pr = P[k, o]
                            if pr == 0.0:
                                continue
                            a = off[k, o, 0]
                            bb = off[k, o, 1]
                            if a > 0:
                                for p in range(NB):
                                    fx[k, p] += (pr * a) * powA[a - 1, p] * powB[bb, p]
                            if bb > 0:
                                for p in range(NB):
                                    fy[k, p] += (pr * bb) * powA[a, p] * powB[bb - 1, p]
                    _chain(T, f, lam, mA, mB, K[st])
                    for d in range(nd):
                        for k in range(2):
                            for p in range(NB):
                                df[k, p] = fx[k, p] * DT[d, 0, p] + fy[k, p] * DT[d, mA, p]
                        if kind[d] == 0:
                            k = rows[d]
                            for o in range(n_out[k]):
                                c = dP[d, o]
                                if c != 0.0:
                                    a = off[k, o, 0]
                                    bb = off[k, o, 1]
                                    for p in range(NB):
                                        df[k, p] += c * powA[a, p] * powB[bb, p]
                        _chain(DT[d], df, lam, mA, mB, DK[st, d])
                        if kind[d] == 1:  # d lambda_k / d log mean_k = -lambda_k
                            k = rows[d]
                            for j in range(start[k], last[k] + 1):
                                for p in range(NB):
                                    DK[st, d, j, p] -= K[st, j, p]
                for r in range(ns):
                    _axpy(G, h * b[r], K[r])
                    _axpy(Dg, h * b[r], DK[r])
            for p in range(nb):
                val, dval = _founder_mix(G, mA, phi, sizes, pis, p)
                out[s, blk + p] = val
                for d in range(nd):
                    dout[s, d, blk + p] = dval * (phi[0] * Dg[d, 0, p] + phi[1] * Dg[d, mA, p])


@njit(cache=True, nogil=True)
def _complement(g, P, off, n_out, k):
    """``h_k = sum_o P[k,o] (1 - (1-g_A)^a (1-g_B)^b)`` and its partial derivatives."""
    la = math.log1p(-g[0]) if g[0] < 1.0 else -np.inf
    lb = math.log1p(-g[1]) if g[1] < 1.0 else -np.inf
    h = 0.0
    hx = 0.0
    hy = 0.0
    for o in range(n_out[k]):
        p = P[k, o]
        a = off[k, o, 0]
        bb = off[k, o, 1]
        if p == 0.0 or a + bb == 0:
            continue
        lm = (a * la if a > 0 else 0.0) + (bb * lb if bb > 0 else 0.0)
        h += p * (-math.expm1(lm) if lm > -np.inf else 1.0)
        if a > 0:
            hx += p * a * (1.0 - g[0]) ** (a - 1) * (1.0 - g[1]) ** bb
        if bb > 0:
            hy += p * bb * (1.0 - g[0]) ** a * (1.0 - g[1]) ** (bb - 1)
    return h, hx, hy


@njit(cache=True, nogil=True)
def _detect_rhs(g, Dg, P, dP, kind, rows, off, n_out, mA, mB, lam, out, dout):
    gg = np.empty(2)
    gg[0] = g[0]
    gg[1] = g[mA]
    start = (0, mA)
    m = (mA, mB)
    nd = Dg.shape[0]
    for k in range(2):
        h, hx, hy = _complement(gg, P, off, n_out, k)
        s0 = start[k]
        for j in range(m[k] - 1):
            out[s0 + j] = lam[k] * (g[s0 + j + 1] - g[s0 + j])
        last = s0 + m[k] - 1
        out[last] = lam[k] * (h - g[last])
        for d in range(nd):
            for j in range(m[k] - 1):
                dout[d, s0 + j] = lam[k] * (Dg[d, s0 + j + 1] - Dg[d, s0 + j])
            dh = hx * Dg[d, 0] + hy * Dg[d, mA]
            if kind[d] == 0 and rows[d] == k:
                la = math.log1p(-gg[0]) if gg[0] < 1.0 else -np.inf
                lb = math.log1p(-gg[1]) if gg[1] < 1.0 else -np.inf
                for o in range(n_out[k]):
                    a = off[k, o, 0]
                    bb = off[k, o, 1]
                    if dP[d, o] == 0.0 or a + bb == 0:
                        continue
                    lm = (a * la if a > 0 else 0.0) + (bb * lb if bb > 0 else 0.0)
                    dh += dP[d, o] * (-math.expm1(lm) if lm > -np.inf else 1.0)
            dout[d, last] = lam[k] * (dh - Dg[d, last])
            if kind[d] == 1 and rows[d] == k:
                for j in range(m[k]):
                    dout[d, s0 + j] -= out[s0 + j]


@njit(cache=True, nogil=True)
def _detect(P, dP, kind, rows, off, n_out, mA, mB, lam, capture, phi, sizes, pis, seg_n,
            seg_dt, A, b):  # fmt: skip
    """``P(clone detected)`` per horizon and its derivatives (complement form)."""
    M = mA + mB
    ns = b.shape[0]
    nd = dP.shape[0]
    H = seg_n.shape[0]
    g = np.empty(M)
    for j in range(mA):
        g[j] = capture[0]
    for j in range(mB):
        g[mA + j] = capture[1]
    Dg = np.zeros((nd, M))
    T = np.empty(M)
    DT = np.empty((nd, M))
    K = np.empty((ns, M))
    DK = np.empty((ns, nd, M))
    q = np.empty(H)
    dq = np.empty((H, nd))
    for s in range(H):
        h = seg_dt[s]
        for _ in range(seg_n[s]):
            for st in range(ns):
                T[:] = g
                DT[:] = Dg
                for r in range(st):
                    a = A[st, r]
                    if a != 0.0:
                        T += (h * a) * K[r]
                        DT += (h * a) * DK[r]
                _detect_rhs(T, DT, P, dP, kind, rows, off, n_out, mA, mB, lam, K[st], DK[st])
            for r in range(ns):
                g += (h * b[r]) * K[r]
                Dg += (h * b[r]) * DK[r]
        sv = phi[0] * g[0] + phi[1] * g[mA]
        ls = math.log1p(-sv) if sv < 1.0 else -np.inf
        qq = 0.0
        dd = 0.0
        for i in range(sizes.shape[0]):
            n = sizes[i]
            qq += pis[i] * (-math.expm1(n * ls) if ls > -np.inf else 1.0)
            dd += pis[i] * n * (1.0 - sv) ** (n - 1)
        q[s] = qq
        for d in range(nd):
            dq[s, d] = dd * (phi[0] * Dg[d, 0] + phi[1] * Dg[d, mA])
    return q, dq


# ---------------------------------------------------------------------- model spec
@dataclass(frozen=True)
class BHModel:
    """Everything the exact BH law needs besides outcome probabilities and lifetimes.

    Attributes:
        offspring: ``(2, O, 2)`` daughters per outcome (from the law).
        n_out: ``(2,)`` outcomes per type.
        phases: ``(2,)`` integer Gamma shapes ``m_k`` (method of stages).
        capture: ``(2,)`` capture probability per true type.
        confusion: ``(2, 2)`` true type -> label probabilities.
        founder: ``(2,)`` founder-type probabilities (fixed).
        sizes: Founder sizes.
        size_probs: Their probabilities (fixed).
        detected: Condition on a non-empty clone (``selection=detected``).
        dt: Runge-Kutta step (time units of the horizons).
        integrator: ``"dop853"`` or ``"rk4"``.
    """

    offspring: np.ndarray
    n_out: np.ndarray
    phases: np.ndarray
    capture: np.ndarray
    confusion: np.ndarray
    founder: np.ndarray
    sizes: np.ndarray
    size_probs: np.ndarray
    detected: bool = True
    dt: float = DEFAULT_DT
    integrator: str = DEFAULT_INTEGRATOR

    @property
    def f0(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coefficients of the per-cell observation pgf ``c0 + cx x + cy y``."""
        c = np.asarray(self.capture, dtype=float)
        C = np.asarray(self.confusion, dtype=float)
        return 1.0 - c, c * C[:, 0], c * C[:, 1]

    def rates(self, mean_lifetimes) -> np.ndarray:
        """Phase rates ``lambda_k = m_k / mean_k``."""
        return self.phases / np.asarray(mean_lifetimes, dtype=float)


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    parts = max(1, min(int(parts), n))
    edges = np.linspace(0, n, parts + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def _run_chunks(fn, n: int, workers: int, pool: ThreadPoolExecutor | None) -> None:
    chunks = _split(n, workers)
    if len(chunks) == 1 or pool is None:
        for a, b in chunks:
            fn(a, b)
        return
    for f in [pool.submit(fn, a, b) for a, b in chunks]:
        f.result()


def law_and_jac(
    model: BHModel,
    P: np.ndarray,
    mean_lifetimes: np.ndarray,
    horizons,
    grid: GridSpec,
    dP: np.ndarray | None = None,
    kind: np.ndarray | None = None,
    rows: np.ndarray | None = None,
    workers: int = 1,
    pool: ThreadPoolExecutor | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Unconditional laws ``p (H, Lx, Ly)`` at every horizon and derivatives ``dp (H, D, Lx, Ly)``.

    ``dP (D, O)``, ``kind (D,)``, ``rows (D,)`` define the derivative directions
    (``None`` = law only, ``dp`` is then ``None``). ``workers`` > 1 splits the grid
    points over ``pool`` (the kernel releases the GIL).
    """
    n, h = segments(horizons, model.dt)
    A, b = tableau(model.integrator)
    Lx, Ly = grid.Lx, grid.Ly
    Ly2 = Ly // 2 + 1
    zx = grid.rx * np.exp(-2j * np.pi * np.arange(Lx) / Lx)
    zy = grid.ry * np.exp(-2j * np.pi * np.arange(Ly2) / Ly)
    npts = Lx * Ly2
    H = len(n)
    lam = model.rates(mean_lifetimes)
    mA, mB = (int(v) for v in model.phases)
    c0, cx, cy = model.f0
    P = np.ascontiguousarray(P, dtype=float)
    common = (model.offspring, model.n_out, mA, mB, lam, c0, cx, cy, model.founder,
              model.sizes, model.size_probs, n, h, A, b)  # fmt: skip
    out = np.empty((H, npts), np.complex128)
    if dP is None or len(dP) == 0:

        def work(i0, i1):
            _law_points(i0, i1, zx, zy, P, *common, out)

        dout = None
    else:
        dP = np.ascontiguousarray(dP, dtype=float)
        kind = np.asarray(kind, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        dout = np.empty((H, len(dP), npts), np.complex128)

        def work(i0, i1):
            _law_points_tan(i0, i1, zx, zy, P, dP, kind, rows, *common, out, dout)

    _run_chunks(work, npts, workers, pool)
    scale = None
    if grid.rx != 1.0 or grid.ry != 1.0:
        scale = grid.rx ** -np.arange(Lx)[:, None] * grid.ry ** -np.arange(Ly)[None, :]
    p = scipy.fft.irfft2(out.reshape(H, Lx, Ly2), s=(Lx, Ly), axes=(1, 2), workers=1)
    if scale is not None:
        p *= scale
    dp = None
    if dout is not None:
        dp = scipy.fft.irfft2(dout.reshape(H, len(dP), Lx, Ly2), s=(Lx, Ly), axes=(2, 3),
                              workers=1)  # fmt: skip
        if scale is not None:
            dp *= scale
    return p, dp


def detection_and_grad(
    model: BHModel,
    P: np.ndarray,
    mean_lifetimes: np.ndarray,
    horizons,
    dP: np.ndarray | None = None,
    kind: np.ndarray | None = None,
    rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``P(clone detected)`` per horizon ``(H,)`` and derivatives ``(H, D)``."""
    n, h = segments(horizons, model.dt)
    A, b = tableau(model.integrator)
    if dP is None:
        dP = np.zeros((0, P.shape[1]))
        kind = rows = np.zeros(0, np.int64)
    mA, mB = (int(v) for v in model.phases)
    return _detect(
        np.ascontiguousarray(P, dtype=float), np.ascontiguousarray(dP, dtype=float),
        np.asarray(kind, np.int64), np.asarray(rows, np.int64), model.offspring, model.n_out,
        mA, mB, model.rates(mean_lifetimes), np.asarray(model.capture, float), model.founder,
        model.sizes, model.size_probs, n, h, A, b,
    )  # fmt: skip


def marginal_laws(model: BHModel, P, mean_lifetimes, horizons, N: int = 4096):
    """Marginal laws of the A and B labels on ``N`` points: ``(pA (H, N), pB (H, N))``.

    The mass a ``Lx x Ly`` table misses is at most ``pA[:, Lx:].sum() + pB[:, Ly:].sum()``.
    """
    pa, _ = law_and_jac(model, P, mean_lifetimes, horizons, GridSpec(N, 2, 1.0, 1.0))
    pb, _ = law_and_jac(model, P, mean_lifetimes, horizons, GridSpec(2, N, 1.0, 1.0))
    return pa.sum(2), pb.sum(1)
