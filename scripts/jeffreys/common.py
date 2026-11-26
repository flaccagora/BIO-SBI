"""Shared helpers of the XG2 Jeffreys-prior study (scripts/jeffreys)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

RESULTS = Path("/home/matteo/BIO-SBI/results/exact_gw")
XG2 = RESULTS / "data_size"
FLAT_BTOA = RESULTS / "data_size_flat_btoa"
JEFF = RESULTS / "data_size_jeffreys"
HORIZONS = (4, 6, 8, 10)


def jeffreys_prior(tail_tol: float = 1e-12):
    """JeffreysPrior of the XG2 model (preset exact_gw_data_size: two_type, perfect, detected)."""
    from hydra import compose, initialize_config_module

    from bio_sbi.inference.gw_fisher import JeffreysPrior
    from bio_sbi.pipeline import build_components

    with initialize_config_module("bio_sbi.conf", version_base=None):
        cfg = compose("config", overrides=["+experiment=exact_gw_data_size", "device=cpu"])
    comp = build_components(cfg)
    horizons = list(cfg.design.horizons)
    return JeffreysPrior(comp.simulator, comp.generator.observation, horizons, tail_tol=tail_tol)


def evaluate(prior, thetas: np.ndarray, device: str = "cuda", chunk: int = 256,
             log=print) -> dict[str, np.ndarray]:  # fmt: skip
    """``log p_J`` (``0.5 log det I``, unnormalised, theta coordinates) and diagnostics."""
    from bio_sbi.inference.gw_fisher import batch_information

    thetas = np.atleast_2d(thetas)
    out = {k: [] for k in ("log_pj", "min_eig", "omitted", "tail", "grid10", "info")}
    t0 = time.time()
    for i in range(0, len(thetas), chunk):
        res = batch_information(prior, thetas[i : i + chunk], device=device, max_batch=16,
                                max_elements=4e7)  # fmt: skip
        for r in res:
            out["log_pj"].append(0.5 * r.logdet)
            out["min_eig"].append(r.min_eigenvalue)
            out["omitted"].append(max(h.omitted_mass for h in r.horizons))
            out["tail"].append(np.nanmax([h.tail_mass for h in r.horizons]))
            out["grid10"].append(r.horizons[-1].grid)
            out["info"].append(r.info)
        if log:
            done = min(i + chunk, len(thetas))
            log(f"  {done}/{len(thetas)} ({(time.time() - t0) / done:.3f} s each)")
    return {k: np.asarray(v) for k, v in out.items()}
