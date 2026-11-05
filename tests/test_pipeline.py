"""End-to-end runs through the Hydra config tree (tiny budgets)."""

import json

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_module

from bio_sbi.collect import collect
from bio_sbi.pipeline import build_components, infer_measured, run_experiment

TINY = [
    "budget=smoke",
    "budget.simulations=128",
    "budget.max_epochs=2",
    "budget.sbc_cases=4",
    "budget.posterior_samples=40",
    "budget.sbc_samples=20",
    "budget.mcmc_chains=4",
    "budget.mcmc_warmup=5",
    "design.clones=6",
    "workers=2",
]


def config(*overrides):
    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        return compose("config", overrides=[*TINY, *overrides], return_hydra_config=False)


@pytest.mark.parametrize("simulator", ["galton_watson", "bellman_harris"])
@pytest.mark.parametrize("law", ["reversible_ab", "hierarchy_ab", "minimal_ab"])
@pytest.mark.parametrize("observation", ["perfect", "noisy", "all_clones"])
def test_all_components_compose(simulator, law, observation):
    comp = build_components(
        config(f"simulator={simulator}", f"law={law}", f"observation={observation}")
    )
    theta = comp.simulator.prior().sample((3,)).numpy()
    x, _ = comp.generator.simulate_theta(theta, "t")
    assert x.shape == (3, comp.generator.x_dim)


def save_hydra_config(cfg, out):
    from omegaconf import OmegaConf

    (out / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, out / ".hydra" / "config.yaml")


@pytest.mark.slow
@pytest.mark.parametrize("method", ["npe", "nle", "nre"])
def test_run_experiment(tmp_path, method):
    cfg = config(f"inference={method}")
    out = tmp_path / method
    save_hydra_config(cfg, out)
    metrics = run_experiment(cfg, out)
    for name in (
        "posterior.pt",
        "metrics.json",
        "recovery.csv",
        "reference.npz",
        "figures/posterior.png",
        "figures/sbc.png",
        "manifest.json",
    ):
        assert (out / name).exists(), name
    assert metrics["calibration"]["cases"] > 0
    assert json.loads((out / "metrics.json").read_text())["inference"] == method

    csv = tmp_path / "measured.csv"
    ref = np.load(out / "reference.npz")
    rows = [(h, *c) for h in (4, 8, 12) for c in ref[f"counts_h{h}"]]
    pd.DataFrame(rows, columns=["horizon", "A", "B"]).to_csv(csv, index=False)
    measured = infer_measured(out, csv, tmp_path / "measured", n=20)
    assert len(pd.read_csv(measured / "posterior_samples.csv.gz")) == 20

    runs, params = collect(tmp_path)
    assert len(runs) == 1 and len(params) == 9


@pytest.mark.slow
def test_sequential_npe_and_bellman_harris(tmp_path):
    cfg = config("simulator=bellman_harris", "inference.method.rounds=2")
    metrics = run_experiment(cfg, tmp_path)
    assert len(metrics["training"]) == 2
    assert isinstance(metrics["calibration"], str)  # SBC skipped for sequential posteriors


@pytest.mark.slow
def test_seed_reproducibility(tmp_path):
    a = run_experiment(config("budget.sbc_cases=0"), tmp_path / "a")
    b = run_experiment(config("budget.sbc_cases=0"), tmp_path / "b")
    assert a["recovery"]["per_parameter"] == b["recovery"]["per_parameter"]
