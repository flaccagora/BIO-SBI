"""Exact derivative of the probability table, toy example.

One type, f(s) = d + p s + q s^2 with d = 1 - p - q (death absorbs the change).
Parameter direction: increase q (divide) at fixed p, i.e. move mass from death to division.
We differentiate P(Z_2 = k) with respect to q three ways:
  (1) forward mode through the iteration, then the same DFT  (what the code does)
  (2) symbolic: expand F_2 = f(f(s)) as a polynomial in s and differentiate in q
  (3) central finite differences
and finally the gradient of the log-likelihood of the toy data.
"""
import numpy as np

np.set_printoptions(precision=6, suppress=True)
p, q, n, L = 0.3, 0.5, 2, 5
w = np.exp(2j * np.pi * np.arange(L) / L)

f = lambda s, q: (1 - p - q) + p * s + q * s**2      # offspring pgf
df_ds = lambda s, q: p + 2 * q * s                    # derivative in s (Jacobian of the map)
df_dq = lambda s, q: s**2 - 1                         # explicit derivative in q (d = 1-p-q moves too)


def table_and_tangent(q):
    """Iterate (value, tangent) together, then DFT both."""
    g = w.copy()              # F_0(s) = s
    t = np.zeros_like(w)      # dF_0/dq = 0
    for _ in range(n):
        g, t = f(g, q), df_dq(g, q) + df_ds(g, q) * t   # chain rule, uses the OLD g
    return np.real(np.fft.fft(g)) / L, np.real(np.fft.fft(t)) / L


P_tab, dP_forward = table_and_tangent(q)

# (2) symbolic: F_2 coefficients as polynomials in q, via numpy polynomial in s for q +- tiny is not symbolic;
#     do it by hand: F_2(s) = d + p f(s) + q f(s)^2, differentiate in q with d = 1-p-q:
Poly = np.polynomial.polynomial
fc = np.array([1 - p - q, p, q])                    # coefficients of f in s
dfc = np.array([-1.0, 0.0, 1.0])                    # d/dq of those coefficients
F2 = Poly.polyadd(Poly.polyadd([1 - p - q], p * fc), q * Poly.polymul(fc, fc))
dF2 = Poly.polyadd(Poly.polyadd([-1.0], p * dfc),
                   Poly.polyadd(Poly.polymul(fc, fc), 2 * q * Poly.polymul(fc, dfc)))

h = 1e-6
dP_fd = (table_and_tangent(q + h)[0] - table_and_tangent(q - h)[0]) / (2 * h)

print("table P(Z2=k)            :", P_tab, "(symbolic:", F2, ")")
print("dP/dq forward mode + DFT :", dP_forward)
print("dP/dq symbolic           :", dF2)
print("dP/dq finite differences :", dP_fd)
print("max |forward - symbolic| :", np.abs(dP_forward - dF2).max())
print("max |forward - FD|       :", np.abs(dP_forward - dP_fd).max())

# score of the toy data (detected sizes 1,2,2,4,1,3)
data = np.array([1, 2, 2, 4, 1, 3])
p0, dp0 = P_tab[0], dP_forward[0]
score = np.sum(dP_forward[data] / P_tab[data] + dp0 / (1 - p0))


def loglik(q):
    P = table_and_tangent(q)[0]
    return np.sum(np.log(P[data] / (1 - P[0])))


print("d logL/dq exact          :", round(score, 6),
      " finite difference:", round((loglik(q + h) - loglik(q - h)) / (2 * h), 6))
