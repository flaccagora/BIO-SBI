"""Theorem 3 check: Var(W) sees the offspring second factorial moments through one scalar.

Also builds a compensated pair (B switching -> B division, offset by fewer A divisions)
with identical mean matrix and identical Var(W), and compares its KL with the
uncompensated move.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gwexact import THETA_STAR, OFF, law, kl_cond


def moments(P):
    M = np.zeros((2, 2)); G = np.zeros((2, 2, 2)); V = np.zeros((2, 2, 2))
    for k in (0, 1):
        for p, (x, y) in zip(P[6 * k:6 * k + 6], OFF[k]):
            v = np.array([x, y], float); M[k] += p * v; V[k] += p * np.outer(v, v)
        G[k] = V[k] - np.diag(M[k]); V[k] -= np.outer(M[k], M[k])
    return M, G, V


def var_w(P, rho, r, J=400):
    M, _, V = moments(P)
    e = np.array([1.0, 0.0]); tot = 0.0; m = e.copy()
    for j in range(J):
        tot += rho ** (-2 * (j + 1)) * sum(m[k] * (r @ V[k] @ r) for k in (0, 1))
        m = m @ M
    return tot


M, G, V = moments(THETA_STAR)
w, R = np.linalg.eig(M); i = int(np.argmax(w.real)); rho = w[i].real
r = R[:, i].real; r = r / r[0]
wl, U = np.linalg.eig(M.T); u = U[:, int(np.argmax(wl.real))].real; u = u / (u @ r)
print(f"rho {rho:.4f}  lambda2 {w[1 - i].real:.4f}  r {r.round(4)}  u {u.round(4)}")
beta = np.zeros(2); m = np.array([1.0, 0.0])
for j in range(400):
    beta += rho ** (-2 * (j + 1)) * m; m = m @ M
print("beta_k (weight of r^T G_k r in Var W):", beta.round(4))
VW = var_w(THETA_STAR, rho, r)
for n, L in ((10, 384), (12, 1024)):
    p = law(THETA_STAR, n, L); a = np.arange(L)
    X = (r[0] * a[:, None] + r[1] * a[None, :]) / rho ** n
    m1 = (p * X).sum()
    print(f"n={n}: exact Var(r.Z_n/rho^n) = {(p * X**2).sum() - m1**2:.4f}   formula Var W = {VW:.4f}")

eps = 0.15
dB = np.zeros(12); dB[6] += eps; dB[8] -= 2 * eps; dB[9] += eps   # r'dG_B r = 2 eps r_A^2
need = -beta[1] * 2 * eps * r[0] ** 2 / beta[0]
eta = -need / (2 * (r[0] - r[1]) ** 2)                             # A: (a-eta, c-eta, b+2eta)
dA = np.zeros(12); dA[3] -= eta; dA[5] -= eta; dA[4] += 2 * eta
print(f"eps {eps}, compensating eta {eta:.4f}")
H = (4, 6, 10); Ls = {4: 17, 6: 65, 10: 384}
star = {n: law(THETA_STAR, n, Ls[n]) for n in H}
for name, P in (("B move only", THETA_STAR + dB), ("B move + A compensation", THETA_STAR + dB + dA)):
    Mx, _, _ = moments(P)
    kl = [kl_cond(star[n], law(P, n, Ls[n])) for n in H]
    print(f"{name:26s} M unchanged: {np.allclose(Mx, M)}  Var W {var_w(P, rho, r):.4f} "
          f"(truth {VW:.4f})  KL/clone n=4,6,10: {np.round(kl, 5)}  B.to_A {P[8]:.2f}")
