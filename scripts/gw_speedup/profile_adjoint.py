"""Time split of one kernel=adjoint gradient call (two_type, gens 4/6/8/10)."""

import copy
import sys
import time

import numpy as np

from bio_sbi.inference import gw_exact as G
from profile_parts import make, t

clones = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
lik, th, _ = make("two_type", (4, 6, 8, 10), clones)
adj = copy.copy(lik)
adj.kernel = "adjoint"
print("total grad ms", t(lambda: adj.log_likelihood(th)))
print("jacobian ms", t(lambda: lik.jacobian(th)))
P, dP, rows = lik.jacobian(th)
m = lik.model
mono = G.monomials(m)
print("monomials+coeffs ms", t(lambda: G.poly_coefficients(G.monomials(m)[2], 6, P, dP, rows)))
C, dC = G.poly_coefficients(mono[2], len(mono[0]), P, dP, rows)
tot = {}
for h in lik.data:
    g = h.grid
    r = {}
    r["value+store"] = t(lambda: G.law_poly(m, C, mono, h.generations, g, True))
    r["value"] = t(lambda: G.law_poly(m, C, mono, h.generations, g, False))
    p, traj = G.law_poly(m, C, mono, h.generations, g, True)
    cot = np.zeros((g.Lx, g.Ly))
    cot[h.a, h.b] = h.weight / p[h.a, h.b]
    r["fft_adjoint"] = t(lambda: G._fft_adjoint(cot, g))
    A = G._fft_adjoint(cot, g)
    barC = np.zeros((2, 6))
    r["backward"] = t(lambda: G._pgf_adjoint(C, mono[0], mono[1], mono[3], h.generations,
                                             m.founder, m.sizes, m.size_probs, traj, A, barC))
    r["detection"] = t(lambda: G.detection_and_grad(m, P, dP, rows, h.generations))
    for k, v in r.items():
        tot[k] = tot.get(k, 0) + v
print({k: round(v, 3) for k, v in tot.items()})
time.sleep(0)
