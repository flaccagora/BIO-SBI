"""Exact endpoint law of the two-type Bellman-Harris simulator (integer Gamma shape).

Model (src/bio_sbi/simulators/bellman_harris.py, ``bh_forward``): every cell of type k
lives Gamma(m, mean_k / m); at the end of its life it draws one outcome from row k and
is replaced by that outcome's daughters, all born at that instant with age 0
(``persist`` = one new same-type cell, i.e. a fresh lifetime; ``to_other`` = one new
cell of the other type; death = nothing). The single founder (type A) is born at t = 0
with age 0. Data = numbers (A, B) of cells alive at the horizon (lifetime end >= t).

Method of stages: Gamma(m, mean/m) = sum of m Exp(lambda_k) phases, lambda_k =
m / mean_k. With G_{k,j}(x, y, t) the pgf of the living population at time t of a
clone started by one type-k cell at the start of phase j, the backward equations are

    dG_{k,j}/dt = lambda_k (G_{k,j+1} - G_{k,j}),           j = 1..m-1
    dG_{k,m}/dt = lambda_k (f_k(G_{A,1}, G_{B,1}) - G_{k,m}),
    G_{k,j}(0) = x (k = A) or y (k = B),

and the clone pgf is F(x, y, t) = G_{A,1}(x, y, t). f_k is row k's offspring pgf in the
gwexact.py ordering (NAMES / OFF there):

    A: death + persist x + to_B y + div_AA x^2 + div_AB x y + div_BB y^2
    B: death + persist y + to_A x + div_AA x^2 + div_AB x y + div_BB y^2

(the repo's B row is ordered death, to_A, persist; ``from_repo`` reorders). Each grid
point (x, y) = (w^a, w^b), w = exp(2 pi i / L), is an independent 2m-dimensional complex
ODE, integrated with a fixed-step explicit Runge-Kutta method (DOP853 tableau, 8th
order, default dt = 1 h; RK4 available) in a numba kernel (2 threads); conjugate
symmetry F(conj x, conj y) = conj F(x, y) halves the grid; the law is
p = real(fft2(F)) / L^2 as in gwexact.py. Forward-mode tangents (12 polynomial-extension
probability directions + d/dlog mean_A, d/dlog mean_B) are integrated alongside
(``law_bh_jac``), giving the derivatives of the RK4 solution exactly.

Non-integer shapes have no finite phase representation; see ``law_bh``.
"""
import math
import os
import sys

import numba
import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import NAMES, OFF  # noqa: E402

numba.set_num_threads(min(2, numba.config.NUMBA_NUM_THREADS))

EXPO = np.array([OFF[0], OFF[1]], dtype=np.int64)  # (2 rows, 6 outcomes, (a, b))
DEFAULT_DT = 1.0  # hours (DOP853, 8th order); see bhexact_validation.md
DEFAULT_METHOD = "dop853"


def from_repo(Prepo):
    """Repo outcome matrix (2, 6) (B row: death, to_A, persist, ...) -> gwexact 12-vector."""
    A, B = np.asarray(Prepo, float)
    return np.r_[A, B[0], B[2], B[1], B[3:]]


@njit(inline="always")
def _poly(P, k, x, y):
    # 6 outcomes with exponents a + b <= 2 (gwexact OFF)
    s = 0j
    for o in range(6):
        a = EXPO_G[k, o, 0]; b = EXPO_G[k, o, 1]
        t = P[6 * k + o] + 0j
        for _ in range(a):
            t *= x
        for _ in range(b):
            t *= y
        s += t
    return s


EXPO_G = EXPO  # numba captures the global as a constant


@njit(inline="always")
def _mono(k, o, x, y):
    a = EXPO_G[k, o, 0]; b = EXPO_G[k, o, 1]
    t = 1.0 + 0j
    for _ in range(a):
        t *= x
    for _ in range(b):
        t *= y
    return t


@njit(inline="always")
def _dpoly(P, k, x, y):
    fx = 0j; fy = 0j
    for o in range(6):
        a = EXPO_G[k, o, 0]; b = EXPO_G[k, o, 1]
        p = P[6 * k + o]
        if a == 1 and b == 0: fx += p
        elif a == 2: fx += 2 * p * x
        elif a == 1 and b == 1: fx += p * y; fy += p * x
        elif a == 0 and b == 1: fy += p
        elif b == 2: fy += 2 * p * y
    return fx, fy


@njit
def _rhs(G, P, lam, m, out):
    fA = _poly(P, 0, G[0, 0], G[1, 0])
    fB = _poly(P, 1, G[0, 0], G[1, 0])
    for k in range(2):
        for j in range(m - 1):
            out[k, j] = lam[k] * (G[k, j + 1] - G[k, j])
    out[0, m - 1] = lam[0] * (fA - G[0, m - 1])
    out[1, m - 1] = lam[1] * (fB - G[1, m - 1])



@njit(parallel=True, fastmath=True)
def _integrate(xs, ys, P, lam, m, seg_n, seg_dt, A, b):
    """Explicit RK with tableau (A, b) on every grid point; F = G_{A,1} after each segment."""
    npts = xs.shape[0]; nseg = seg_n.shape[0]; ns = b.shape[0]
    out = np.empty((nseg, npts), np.complex128)
    for i in prange(npts):
        G = np.empty((2, m), np.complex128)
        for j in range(m):
            G[0, j] = xs[i]; G[1, j] = ys[i]
        K = np.empty((ns, 2, m), np.complex128)
        T = np.empty_like(G)
        for s in range(nseg):
            h = seg_dt[s]
            for _ in range(seg_n[s]):
                for st in range(ns):
                    for a in range(2):
                        for c in range(m):
                            acc = G[a, c]
                            for r in range(st):
                                if A[st, r] != 0.0:
                                    acc += h * A[st, r] * K[r, a, c]
                            T[a, c] = acc
                    _rhs(T, P, lam, m, K[st])
                for a in range(2):
                    for c in range(m):
                        acc = 0j
                        for r in range(ns):
                            acc += b[r] * K[r, a, c]
                        G[a, c] += h * acc
            out[s, i] = G[0, 0]
    return out

@njit
def _rhs_tan(G, D, P, lam, m, out, dout):
    """dout[d] = directional derivative of the RHS; d = 0..11 P entries, 12, 13 log-means."""
    x = G[0, 0]; y = G[1, 0]
    f = (_poly(P, 0, x, y), _poly(P, 1, x, y))
    for k in range(2):
        for j in range(m - 1):
            out[k, j] = lam[k] * (G[k, j + 1] - G[k, j])
        out[k, m - 1] = lam[k] * (f[k] - G[k, m - 1])
    jac = (_dpoly(P, 0, x, y), _dpoly(P, 1, x, y))
    nd = D.shape[0]
    for d in range(nd):
        for k in range(2):
            for j in range(m - 1):
                dout[d, k, j] = lam[k] * (D[d, k, j + 1] - D[d, k, j])
            fx, fy = jac[k]
            df = fx * D[d, 0, 0] + fy * D[d, 1, 0]
            if d < 12 and d // 6 == k:
                df += _mono(k, d % 6, x, y)
            dout[d, k, m - 1] = lam[k] * (df - D[d, k, m - 1])
        if d >= 12:  # d lambda_k / d log mean_k = -lambda_k
            k = d - 12
            for j in range(m):
                dout[d, k, j] -= out[k, j]



@njit(parallel=True, fastmath=True)
def _integrate_tan(xs, ys, P, lam, m, seg_n, seg_dt, A, b):
    npts = xs.shape[0]; nseg = seg_n.shape[0]; nd = 14; ns = b.shape[0]
    out = np.empty((nseg, npts), np.complex128)
    dout = np.empty((nseg, nd, npts), np.complex128)
    for i in prange(npts):
        G = np.empty((2, m), np.complex128)
        for j in range(m):
            G[0, j] = xs[i]; G[1, j] = ys[i]
        D = np.zeros((nd, 2, m), np.complex128)
        K = np.empty((ns, 2, m), np.complex128)
        DK = np.empty((ns, nd, 2, m), np.complex128)
        T = np.empty_like(G); DT = np.empty_like(D)
        for s in range(nseg):
            h = seg_dt[s]
            for _ in range(seg_n[s]):
                for st in range(ns):
                    T[:] = G
                    DT[:] = D
                    for r in range(st):
                        if A[st, r] != 0.0:
                            c = h * A[st, r]
                            T += c * K[r]
                            DT += c * DK[r]
                    _rhs_tan(T, DT, P, lam, m, K[st], DK[st])
                for r in range(ns):
                    G += (h * b[r]) * K[r]
                    D += (h * b[r]) * DK[r]
            out[s, i] = G[0, 0]
            dout[s, :, i] = D[:, 0, 0]
    return out, dout

def _segments(horizons, dt):
    hs = np.asarray(horizons, float)
    if np.any(np.diff(hs) <= 0) or hs[0] <= 0:
        raise ValueError("horizons must be positive and increasing")
    lens = np.diff(np.r_[0.0, hs])
    n = np.maximum(1, np.ceil(lens / dt - 1e-9)).astype(np.int64)
    return n, lens / n


def _check(P, mean_lifetimes, shape):
    if abs(shape - round(shape)) > 1e-12 or shape < 1:
        raise NotImplementedError(
            "Non-integer Gamma shape: the method of stages needs an integer shape. Route: "
            "solve the general Bellman-Harris integral equation "
            "F_k(t) = s_k (1 - G_k(t)) + int_0^t f_k(F(t - u)) dG_k(u) "
            "(s_k = x or y, G_k the lifetime cdf) on the grid, e.g. by a product-"
            "integration / convolution scheme in t.")
    P = np.asarray(P, float)
    if P.shape != (12,):
        raise ValueError("P must be the 12-vector in gwexact NAMES order")
    m = int(round(shape))
    lam = m / np.asarray(mean_lifetimes, float)
    return P, m, lam


def _tableau(method):
    if method == "rk4":
        A = np.zeros((4, 4)); A[1, 0] = A[2, 1] = 0.5; A[3, 2] = 1.0
        return A, np.array([1, 2, 2, 1]) / 6
    if method == "dop853":
        from scipy.integrate._ivp import dop853_coefficients as c
        return np.ascontiguousarray(c.A[:12, :12]), np.ascontiguousarray(c.B)
    raise ValueError(method)


def _dims(L):
    return (int(L), int(L)) if np.ndim(L) == 0 else (int(L[0]), int(L[1]))


def _half_grid(LA, LB):
    """Grid points with A index a = 0..LA//2 (the rest follow from F(conj) = conj F)."""
    wa = np.exp(2j * np.pi * np.arange(LA // 2 + 1) / LA)
    wb = np.exp(2j * np.pi * np.arange(LB) / LB)
    x, y = np.meshgrid(wa, wb, indexing="ij")  # axis 0 = A, axis 1 = B
    return x.ravel(), y.ravel()


def _full(Fh, LA, LB):
    """(..., LA//2+1, LB) half grid -> (..., LA, LB) via F(w^-a, w^-b) = conj F(w^a, w^b)."""
    F = np.empty(Fh.shape[:-2] + (LA, LB), complex)
    F[..., : LA // 2 + 1, :] = Fh
    a = np.arange(LA // 2 + 1, LA)
    F[..., a, :] = np.conj(Fh[..., (LA - a), :][..., (-np.arange(LB)) % LB])
    return F


def law_bh(P, mean_lifetimes, shape, horizon_h, L, dt=DEFAULT_DT, method=DEFAULT_METHOD):
    """Joint law of (A, B) living at ``horizon_h`` (scalar -> (LA, LB); list -> (H, LA, LB)).

    ``L`` = int (square table) or (LA, LB).
    Unconditional (p[0, 0] = extinction); condition with ``condition``. Mass of counts
    >= L is aliased onto the table (check with ``marginal_tails``).
    """
    P, m, lam = _check(P, mean_lifetimes, shape)
    scalar = np.ndim(horizon_h) == 0
    n, h = _segments(np.atleast_1d(horizon_h), dt)
    LA, LB = _dims(L)
    xs, ys = _half_grid(LA, LB)
    F = _integrate(xs, ys, P, lam, m, n, h, *_tableau(method))
    F = _full(F.reshape(len(n), LA // 2 + 1, LB), LA, LB)
    p = np.real(np.fft.fft2(F, axes=(1, 2))) / (LA * LB)
    return p[0] if scalar else p


def law_bh_jac(P, mean_lifetimes, shape, horizons, L, dt=DEFAULT_DT, method=DEFAULT_METHOD):
    """p (H, LA, LB) and dp (H, 14, LA, LB): d/dP_i (12, polynomial extension, gwexact
    order) and d/dlog mean_A, d/dlog mean_B."""
    P, m, lam = _check(P, mean_lifetimes, shape)
    n, h = _segments(np.atleast_1d(horizons), dt)
    LA, LB = _dims(L)
    xs, ys = _half_grid(LA, LB)
    F, dF = _integrate_tan(xs, ys, P, lam, m, n, h, *_tableau(method))
    H = len(n)
    F = _full(F.reshape(H, LA // 2 + 1, LB), LA, LB)
    dF = _full(dF.reshape(H, 14, LA // 2 + 1, LB), LA, LB)
    p = np.real(np.fft.fft2(F, axes=(1, 2))) / (LA * LB)
    dp = np.real(np.fft.fft2(dF, axes=(2, 3))) / (LA * LB)
    return p, dp


def marginal_tails(P, mean_lifetimes, shape, horizons, N=8192, dt=DEFAULT_DT,
                   method=DEFAULT_METHOD):
    """Marginal laws of A (y = 1) and B (x = 1) on N points: (pA, pB), each (H, N).

    P(A >= L or B >= L) <= tailA(L) + tailB(L) bounds the mass a table of size L misses.
    """
    P, m, lam = _check(P, mean_lifetimes, shape)
    n, h = _segments(np.atleast_1d(horizons), dt)
    w = np.exp(2j * np.pi * np.arange(N) / N); one = np.ones(N, complex)
    tab = _tableau(method)
    FA = _integrate(w, one, P, lam, m, n, h, *tab)
    FB = _integrate(one, w, P, lam, m, n, h, *tab)
    return np.real(np.fft.fft(FA, axis=1)) / N, np.real(np.fft.fft(FB, axis=1)) / N


def condition(p, floor=1e-15):
    """Law conditioned on detection (A + B > 0); floored for logs."""
    q = p / (1 - p[..., :1, :1])
    q = q.copy()
    q[..., 0, 0] = 0
    return np.clip(q, floor, None)


def log_lik(counts, horizons, P, mean_lifetimes, shape, L, dt=DEFAULT_DT, method=DEFAULT_METHOD):
    """Sum of log P(A, B | detected) over clones; counts[h] = (n, 2) int array."""
    p = law_bh(P, mean_lifetimes, shape, list(horizons), L, dt, method)
    ll = 0.0
    for q, h in zip(condition(p), horizons):
        c = np.asarray(counts[h])
        if c.size and (c[:, 0].max() >= p.shape[1] or c[:, 1].max() >= p.shape[2]):
            return -np.inf
        ll += np.log(q[c[:, 0], c[:, 1]]).sum()
    return ll


__all__ = ["law_bh", "law_bh_jac", "marginal_tails", "condition", "log_lik", "from_repo",
           "NAMES"]
