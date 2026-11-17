"""Where does a gradient call of the exact GW likelihood spend its time?

    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 uv run python scripts/gw_speedup/profile_parts.py 128
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft
import yaml

from bio_sbi.inference import gw_exact as G
from bio_sbi.inference.exact_gw import ExactLikelihood
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import GaltonWatson
from bio_sbi.summaries import Cohort

ROOT = Path(__file__).resolve().parents[2]


def law(name):
    return yaml.safe_load((ROOT / "src/bio_sbi/conf/law" / f"{name}.yaml").read_text())


def make(name="two_type", gens=(4, 6, 8, 10), clones=128, seed=0, obs=None, **kw):
    spec = law(name)
    sim = GaltonWatson(spec)
    truth = sim.physical_from_mapping(spec["reference"])
    rng = np.random.default_rng(seed)
    obs = obs or ObservationModel(sim.types)
    cohorts = [
        Cohort(h, obs.measure(sim.simulate(truth, h, 4 * clones, rng), rng)) for h in gens
    ]
    cohorts = [Cohort(c.horizon, c.counts[c.counts.sum(1) > 0][:clones]) for c in cohorts]
    lik = ExactLikelihood(sim, obs, cohorts, **kw)
    return lik, sim.encode(truth).astype(float), sim


def t(f, n=20):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3


if __name__ == "__main__":
    clones = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    lik, th, _ = make(clones=clones)
    print("grids", [(h.generations, h.grid.Lx, h.grid.Ly, len(h.a)) for h in lik.data])
    print("obj ms", t(lambda: lik.log_likelihood(th, False)))
    print("grad ms", t(lambda: lik.log_likelihood(th)))
    P, dP, rows = lik.jacobian(th)
    m = lik.model
    tot = {}
    for h in lik.data:
        g = h.grid
        Lx, Ly = g.Lx, g.Ly
        Ly2 = Ly // 2 + 1
        zx = g.rx * np.exp(-2j * np.pi * np.arange(Lx) / Lx)
        zy = g.ry * np.exp(-2j * np.pi * np.arange(Ly2) / Ly)
        out = np.empty((Lx, Ly2), complex)
        dout = np.empty((len(dP), Lx, Ly2), complex)
        c0, cx, cy = m.f0

        def k(D, h=h, zx=zx, zy=zy, out=out, dout=dout):
            G._pgf_grid(P, dP[:D], rows[:D], m.offspring, m.n_out, h.generations, zx, zy,
                        c0, cx, cy, m.founder, m.sizes, m.size_probs, out, dout[:D])  # fmt: skip

        r = {
            "pgf0": t(lambda: k(0)),
            "pgfD": t(lambda: k(len(dP))),
            "irfft": t(lambda: scipy.fft.irfft2(out, s=(Lx, Ly), workers=1)),
            "irfftD": t(lambda: scipy.fft.irfft2(dout, s=(Lx, Ly), axes=(1, 2), workers=1)),
            "lawjac": t(lambda: G.law_and_jac(m, P, dP, rows, h.generations, g)),
        }
        print(h.generations, (Lx, Ly), {k_: round(v, 3) for k_, v in r.items()})
        for k_, v in r.items():
            tot[k_] = tot.get(k_, 0) + v
    print("total", {k_: round(v, 3) for k_, v in tot.items()})
