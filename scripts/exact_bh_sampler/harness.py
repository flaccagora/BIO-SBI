"""Exact BH likelihood of a stored reference dataset (sampler benchmarks; run from the repo root).

Names: rev0 / rev3 / two0 / ridge0 = first-look AMIS runs under results/exact_bh (their
reference.npz and .hydra config); small<seed> = {96,144} h x 64, mid<seed> = {120,192} h
x 100, reversible_ab (simulated).
Reparam study (2026-11-11): run_budget.sh (fixed budget, 2 threads), summarise.py
(ESS per 1k evaluations, z vs reference), imh_probe.py (independence acceptance).
Validation (2026-11-10): reference_mix.py small0 <pilot> ref.npz 32 4000 16 (128k evals),
drive.py small0 out.json max_evals=300000, compare.py small0 ref.npz out.npz.
"""

import glob

import numpy as np
from omegaconf import OmegaConf

from bio_sbi.inference.exact_bh import BHLikelihood
from bio_sbi.pipeline import build_components
from bio_sbi.summaries import Cohort

RUNS = {
    "rev0": "results/exact_bh/real_like_rev_a/0_*seed=0",
    "rev3": "results/exact_bh/real_like_rev_b/0_*seed=3",
    "two0": "results/exact_bh/real_like_two_a/0_*seed=0",
    "ridge0": "results/exact_bh/ridge_512_a/0_*seed=0",
}


def small(seed=0, horizons="[96,144]", clones=64, law="reversible_ab"):
    from hydra import compose, initialize_config_module

    from bio_sbi.pipeline import build_data_generator, reference_parameters

    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "+experiment=exact_bh_real_like",
                f"law={law}",
                f"seed={seed}",
                f"design.horizons={horizons}",
                f"design.clones={clones}",
            ],
        )
    comp = build_components(cfg)
    dg = build_data_generator(cfg, comp.generator)
    truth = reference_parameters(cfg, dg.simulator, 0)
    return cfg, comp, dg.dataset(truth, "reference", stats=[])


def load(name, dt=2.0, threads=1, grid=None, **kw):
    if name.startswith("small"):
        cfg, comp, cohorts = small(seed=int(name[5:] or 0))
        d = None
    elif name.startswith("mid"):
        cfg, comp, cohorts = small(seed=int(name[3:] or 0), horizons="[120,192]", clones=100)
        d = None
    else:
        d = glob.glob(RUNS[name])[0]
        cfg = OmegaConf.load(d + "/.hydra/config.yaml")
        comp = build_components(cfg)
        z = np.load(d + "/reference.npz")
        cohorts = [
            Cohort(float(k[len("counts_h") :]), z[k]) for k in z.files if k.startswith("counts_h")
        ]
    cohorts.sort(key=lambda c: c.horizon)
    g = dict(OmegaConf.to_container(cfg.inference.method.grid))
    g.update(grid or {})
    lik = BHLikelihood(
        comp.simulator, comp.generator.observation, cohorts, g, dt=dt, threads=threads, **kw
    )
    return lik, comp, cfg, d
