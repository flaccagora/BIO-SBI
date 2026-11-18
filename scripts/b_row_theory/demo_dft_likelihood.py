"""Smallest end-to-end example: PGF -> values on roots of unity -> DFT -> probability table -> likelihood.

One cell type, offspring pgf f(s) = 0.2 + 0.3 s + 0.5 s^2 (die / persist / divide), two generations.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
Poly = np.polynomial.polynomial


def table(f, n, L):
    """Model's probability table P(Z_n = k), k = 0..L-1, via iteration + DFT."""
    w = np.exp(2j * np.pi * np.arange(L) / L)      # the L roots of unity
    g = w.copy()
    for _ in range(n):                              # F_n = f(f(...f(s))) evaluated pointwise
        g = f(g)
    return w, g, np.real(np.fft.fft(g)) / L         # DFT -> coefficients


f = lambda s: 0.2 + 0.3 * s + 0.5 * s**2
coef = np.array([0.2, 0.3, 0.5])
exact = Poly.polyadd(Poly.polyadd([0.2], 0.3 * coef), 0.5 * Poly.polymul(coef, coef))
print("exact P(Z2=k) by symbolic expansion, k=0..4:", exact)

w, g, c = table(f, n=2, L=5)
print("grid points (roots of unity):", np.round(w, 3))
print("F_2 at the grid points      :", np.round(g, 4))
print("DFT of those values         :", c)

data = np.array([1, 2, 2, 4, 1, 3])                 # observed (detected) clone sizes
q = c / (1 - c[0]); q[0] = 0                         # condition on detection
print("conditioned table           :", q)
print("look-ups for the data       :", q[data], " log-likelihood =", round(np.log(q[data]).sum(), 3))

f2 = lambda s: 0.3 + 0.3 * s + 0.4 * s**2            # a different parameter value
_, _, c2 = table(f2, n=2, L=5)
q2 = c2 / (1 - c2[0]); q2[0] = 0
print("theta2 table                :", c2, " log-likelihood =", round(np.log(q2[data]).sum(), 3))
