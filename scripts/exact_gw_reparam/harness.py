"""XG2 cells for the exact_gw reparametrisation benchmark (run from the repo root).

``load(cell)`` rebuilds the dataset of an XG2 cell with the pipeline (``+experiment=
exact_gw_data_size gt=... design.clones=... seed=... inference.method.kernel=adjoint``),
checks it against the stored ``reference.npz`` of the XG2 run (read-only) and returns
``(ExactGW method, ExactLikelihood, cfg, truth)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_module

from bio_sbi.pipeline import (
    build_components,
    build_data_generator,
    reference_parameters,
    truth_plausible,
)

XG2 = Path("/home/matteo/BIO-SBI/results/exact_gw/data_size")
T5 = ("+truths.T5_C7={source:reference,seed:null,values:{A:{death:0.05,persist:0.10,"
      "to_B:0.10,div_AA:0.50,div_AB:0.15,div_BB:0.10},B:{death:0.08,persist:0.05,to_A:0.10,"
      "div_AA:0.05,div_AB:0.05,div_BB:0.67}}}")  # fmt: skip
#: cell name -> (truth, clones per endpoint, data seed)
CELLS = {
    "T1_n1024_s0": ("T1_ref", 1024, 0),
    "P3_n1024_s0": ("P3", 1024, 0),
    "T3_n512_s1": ("T3_Bdiv", 512, 1),
    "T5_n1024_s0": ("T5_C7", 1024, 0),
    "T1_n128_s0": ("T1_ref", 128, 0),
}


def overrides(gt: str, n: int, seed: int, extra=()) -> list[str]:
    """Hydra overrides of one XG2 cell."""
    out = ["+experiment=exact_gw_data_size", f"gt={gt}", f"design.clones={n}", f"seed={seed}",
           "inference.method.kernel=adjoint"]  # fmt: skip
    if gt == "T5_C7":
        out.insert(1, T5)
    return out + list(extra)


def load(cell: str, extra=(), check: bool = True):
    """Return method, likelihood, config and truth of ``cell`` (dataset checked against XG2)."""
    gt, n, seed = CELLS[cell]
    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        cfg = compose("config", overrides=overrides(gt, n, seed, extra))
    comp = build_components(cfg)
    data_gen = build_data_generator(cfg, comp.generator)
    max_draws = int(cfg.truth.get("max_draws", 50)) if cfg.truth.source == "prior" else 1
    reference = None
    for attempt in range(max_draws):
        truth = reference_parameters(cfg, data_gen.simulator, attempt)
        if not truth_plausible(cfg, data_gen.simulator, truth):
            continue
        reference = data_gen.dataset(truth, "reference", stats=[])
        if reference is not None:
            break
    assert reference is not None
    stored = XG2 / gt / f"n{n}_s{seed}" / "reference.npz"
    if check and stored.exists():
        z = np.load(stored)
        np.testing.assert_allclose(z["truth"], truth, rtol=0, atol=1e-12)
        for c in reference:
            assert np.array_equal(z[f"counts_h{c.horizon:g}"], c.counts), (cell, c.horizon)
    elif check:
        print(f"warning: {stored} not found, dataset not checked")
    lik = comp.method.likelihood(comp.simulator, comp.generator.observation, reference)
    return comp.method, lik, cfg, truth
