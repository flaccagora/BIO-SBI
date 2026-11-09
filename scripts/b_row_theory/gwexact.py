"""Exact likelihood and Fisher information of the two-type GW endpoint law.

Outcome order per row: death, persist, to_other, div_AA, div_AB, div_BB.
Offspring (nA, nB):
  A: (0,0) (1,0) (0,1) (2,0) (1,1) (0,2)
  B: (0,0) (0,1) (1,0) (2,0) (1,1) (0,2)
Data: (A,B) counts at generation n from one A founder, conditioned on Z_n != 0.
"""
import numpy as np

OFF = {
    0: [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)],
    1: [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2)],
}
NAMES = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
         "B.death", "B.persist", "B.to_A", "B.div_AA", "B.div_AB", "B.div_BB"]

THETA_STAR = np.array([.03, .03, .05, .55, .05, .29, .10, .10, .80, 0, 0, 0])


def f_row(P, k, x, y):
    out = 0
    for p, (a, b) in zip(P[6 * k:6 * k + 6], OFF[k]):
        if p != 0:
            out = out + p * x**a * y**b
    return out


def df_row(P, k, x, y):
    """(df/dx, df/dy) for row k."""
    dx = 0; dy = 0
    for p, (a, b) in zip(P[6 * k:6 * k + 6], OFF[k]):
        if p == 0:
            continue
        if a: dx = dx + p * a * x**(a - 1) * y**b
        if b: dy = dy + p * b * x**a * y**(b - 1)
    return dx, dy


def grid(L):
    w = np.exp(2j * np.pi * np.arange(L) / L)
    return np.meshgrid(w, w, indexing="ij")  # x varies on axis 0 (A), y on axis 1 (B)


def pgf_iter(P, n, L, direction=None):
    """Return F_n^A on the grid and, optionally, its directional derivative."""
    x, y = grid(L)
    FA, FB = x.copy(), y.copy()
    dA = np.zeros_like(x) if direction is not None else None
    dB = np.zeros_like(x) if direction is not None else None
    for _ in range(n):
        if direction is not None:
            # chain rule: d f_k(F; P) = (∂_P f_k)[dir](F) + J_f(F) dF
            ax, ay = df_row(P, 0, FA, FB)
            bx, by = df_row(P, 1, FA, FB)
            pA = f_row(direction, 0, FA, FB) if np.any(direction[:6]) else 0
            pB = f_row(direction, 1, FA, FB) if np.any(direction[6:]) else 0
            dA, dB = pA + ax * dA + ay * dB, pB + bx * dA + by * dB
        FA, FB = f_row(P, 0, FA, FB), f_row(P, 1, FA, FB)
    return FA, dA


def probs(F, L):
    # F(x,y) = sum p_ab x^a y^b, x = w^j  -> p = ifft2(F) with conj convention
    return np.real(np.fft.fft2(F)) / L**2 if False else np.real(np.fft.ifft2(F))


def law(P, n, L=None):
    L = L or (2**n + 1)
    FA, _ = pgf_iter(P, n, L)
    p = np.real(np.fft.fft2(FA)) / L**2
    # ifft2 of values at w^j gives coefficients p[a,b]: check sign convention
    return p


def law_and_grads(P, n, directions, L=None):
    L = L or (2**n + 1)
    FA, _ = pgf_iter(P, n, L)
    p = np.real(np.fft.fft2(FA)) / L**2
    grads = []
    for d in directions:
        _, dA = pgf_iter(P, n, L, direction=np.asarray(d, float))
        grads.append(np.real(np.fft.fft2(dA)) / L**2)
    return p, np.array(grads)


def conditional_fisher(p, g, tol=1e-13):
    """Fisher info per clone for the law conditioned on Z != 0."""
    p0, g0 = p[0, 0], g[:, 0, 0]
    mask = p > tol
    mask[0, 0] = False
    pz = p[mask]
    sc = g[:, mask] / pz + (g0 / (1 - p0))[:, None]  # score of conditional law
    pc = pz / (1 - p0)
    I = (sc * pc) @ sc.T
    return I, 1 - pc.sum()  # missed conditional mass


def cond_logp(p, tol=1e-300):
    p0 = p[0, 0]
    q = p / (1 - p0)
    q[0, 0] = 0
    return np.log(np.clip(q, tol, None)), q


def kl_cond(p_true, p_alt):
    lt, qt = cond_logp(p_true)
    la, _ = cond_logp(p_alt)
    m = qt > 1e-15
    return float(np.sum(qt[m] * (lt[m] - la[m])))


def tangent_basis():
    """10 directions spanning the product of the two simplex tangent spaces.

    Basis vector for entry j != death of row k: +1 at entry j, -1 at death.
    """
    T = []
    for k in (0, 1):
        for j in range(1, 6):
            v = np.zeros(12); v[6 * k + j] = 1; v[6 * k] = -1; T.append(v)
    return np.array(T)


def law_damped(P, n, L, r=None):
    """Law on an L x L grid using radius-r evaluation to suppress aliasing (L < 2^n + 1)."""
    if L >= 2**n + 1:
        return law(P, n, L)
    r = r or np.exp(np.log(1e-12) / L)  # aliased mass scaled by <= 1e-12
    x, y = grid(L)
    FA, FB = r * x, r * y
    for _ in range(n):
        FA, FB = f_row(P, 0, FA, FB), f_row(P, 1, FA, FB)
    p = np.real(np.fft.fft2(FA)) / L**2
    a = np.arange(L)
    return p / r ** (a[:, None] + a[None, :])
