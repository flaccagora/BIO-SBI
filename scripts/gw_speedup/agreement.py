"""Agreement of kernel=adjoint with kernel=loop (values and gradients) on random points.

The FFT gives each cell probability to an absolute accuracy of ~eps * r^-(a+b) (round-off
amplified by the damping radius, see gw_exact.grid_spec). A cell with ``p`` near that
level has a round-off-dominated relative error, and its ``w / p * dp`` gradient term is
noise in *both* kernels. Each point is therefore tagged with

    kappa = max over observed (unfloored) cells of eps * r^-(a+b) / p(a, b),

the estimated worst relative round-off of a cell probability, and agreement is reported
per kappa class. Points: the truth, logit-normal perturbations of it, uniform random
points and points with coordinates at 1e-6 / 1e-3 / 1-1e-3 / 1-1e-6.

    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 uv run python scripts/gw_speedup/agreement.py \
        > scripts/gw_speedup/agreement.txt
"""

import copy

import numpy as np
from scipy.special import expit, logit

from bio_sbi.inference import gw_exact as G
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import GaltonWatson
from profile_parts import law, make

CASES = [
    ("two_type", (4, 6, 8, 10), 128, None),
    ("two_type", (4, 6, 10), 1024, None),
    ("reversible_ab", (4, 6), 128, None),
    ("reversible_ab", (3, 6, 9), 256, "confusion"),
    ("two_type", (2, 5), 64, "all"),
]
EPS = np.finfo(float).eps


def observation(sim, kind):
    if kind == "confusion":
        return ObservationModel(sim.types, capture=[0.7, 0.9], confusion=[[0.9, 0.1], [0.2, 0.8]])
    if kind == "all":
        return ObservationModel(sim.types, selection="all", capture=[0.8, 0.8])
    return None


def points(rng, th, n):
    dim = len(th)
    near = expit(logit(th)[None] + rng.normal(0, 0.5, size=(n, dim)))
    uni = rng.uniform(0.02, 0.98, size=(n, dim))
    edge = expit(logit(th)[None] + rng.normal(0, 0.5, size=(n, dim)))
    mask = rng.random((n, dim)) < 0.3
    edge[mask] = rng.choice([1e-6, 1e-3, 1 - 1e-3, 1 - 1e-6], size=mask.sum())
    return {"truth": th[None], "near": near, "uniform": uni, "boundary": edge}


def kappa(lik, theta):
    P = lik.probabilities(theta)
    worst = 0.0
    for h in lik.data:
        g = h.grid
        p = G.law_and_jac(lik.model, P, np.zeros((0, P.shape[1])), np.zeros(0, np.int64),
                          h.generations, g)[0][h.a, h.b]  # fmt: skip
        ok = p > lik.floor
        amp = g.rx ** -h.a.astype(float) * g.ry ** -h.b.astype(float)
        if ok.any():
            worst = max(worst, float((EPS * amp[ok] / p[ok]).max()))
    return worst


def main():
    rng = np.random.default_rng(0)
    rows = []
    for name, gens, clones, kind in CASES:
        sim = GaltonWatson(law(name))
        lik, th, _ = make(name, gens, clones, obs=observation(sim, kind))
        adj = copy.copy(lik)
        adj.kernel = "adjoint"
        for cls, thetas in points(rng, th, 20).items():
            for t in thetas:
                v0, g0 = lik.log_likelihood(t)
                v1, g1 = adj.log_likelihood(t)
                assert lik.log_likelihood(t, False)[0] == v0
                assert adj.log_likelihood(t, False)[0] == v1
                rows.append((name, gens, clones, kind or "perfect", cls, kappa(lik, t),
                             abs(v1 - v0) / max(abs(v0), 1.0),
                             np.abs(g1 - g0).max() / max(np.abs(g0).max(), 1e-300)))  # fmt: skip
    print("case | point class | n | kappa range | max value rel diff | max grad rel diff "
          "(max_d |g1-g0| / max_d |g0|)")
    keys = sorted({r[:5] for r in rows}, key=lambda k: [str(x) for x in k])
    for key in keys:
        sel = [r for r in rows if r[:5] == key]
        ks = np.array([r[5] for r in sel])
        print(f"{key[0]} gens={key[1]} clones={key[2]} obs={key[3]} | {key[4]} | {len(sel)} | "
              f"{ks.min():.1e}..{ks.max():.1e} | {max(r[6] for r in sel):.1e} | "
              f"{max(r[7] for r in sel):.1e}")  # fmt: skip
    print()
    ks = np.array([r[5] for r in rows])
    dg = np.array([r[7] for r in rows])
    for lo, hi in [(0, 1e-8), (1e-8, 1e-6), (1e-6, 1e-4), (1e-4, 1e-2), (1e-2, np.inf)]:
        m = (ks >= lo) & (ks < hi)
        if m.any():
            print(f"kappa in [{lo:.0e}, {hi:.0e}): {m.sum():3d} points, grad rel diff max "
                  f"{dg[m].max():.1e}, median {np.median(dg[m]):.1e}")  # fmt: skip
    print(f"all points: value rel diff max {max(r[6] for r in rows):.1e}")


if __name__ == "__main__":
    main()
