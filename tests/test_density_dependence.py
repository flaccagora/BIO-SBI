"""Density-dependent Bellman-Harris lifetimes (``simulator=bellman_harris_dd``)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from bio_sbi.pipeline import build_components, build_data_generator, reference_parameters
from bio_sbi.simulators import BellmanHarris
from bio_sbi.simulators.bellman_harris import bh_density_forward

from .conftest import load_law

DD = {"capacity_bounds": [1.0, 1e13]}
REAL_2D = Path(__file__).resolve().parents[1] / "data" / "real" / "crc0322_2d_bh.csv"


def make_bh(spec, dd=DD, **kw):
    return BellmanHarris(spec, lifetime_shape=kw.pop("shape", 8.0),
                         lifetime_mean_bounds=[10, 80], density_dependence=dd, **kw)  # fmt: skip


def values(spec, sim, capacity=None, kappa=None):
    out = {**spec["reference"],
           **{f"{t}.mean_lifetime": 20.0 + 5 * i for i, t in enumerate(sim.types)}}  # fmt: skip
    if capacity is not None:
        out["density.capacity"] = capacity
    if kappa is not None:
        out["clone.kappa"] = kappa
    return out


def totals(counts):
    t = counts.sum(1)
    return t, np.log1p(t[t > 0]).std()


# ------------------------------------------------------------------ parameters
def test_off_by_default_and_parameter_order():
    law = load_law()
    off = make_bh(law, dd=None)
    assert not off.density_dependent and "density.capacity" not in off.parameter_names
    assert np.isinf(off.capacity(off.physical_from_mapping(values(law, off))))
    het = load_law("two_type_hetero")
    bh = make_bh(het)
    assert bh.parameter_names[-4:] == (
        "A.mean_lifetime",
        "B.mean_lifetime",
        "density.capacity",
        "clone.kappa",
    )
    assert bh.theta_names[-2:] == ("density.log_capacity", "clone.log_kappa")
    p = bh.physical_from_mapping(values(het, bh, capacity=300.0, kappa=7.0))
    assert np.allclose(bh.mean_lifetimes(p), [20.0, 25.0])
    assert bh.capacity(p) == 300.0 and bh.clone_kappa(p) == 7.0
    assert np.allclose(bh.scalar_values(p), [20.0, 25.0, 300.0])
    assert make_bh(law, dd=True).space.scalars[-1].high == 1e5


def test_default_output_is_bit_identical():
    # Hashes recorded with the code before density dependence existed.
    expected = {"two_type": "a19887e64e355416", "two_type_hetero": "2173e4d3293a65a6",
                "multi_founder": "4229336591de52eb"}  # fmt: skip
    for name, digest in expected.items():
        law = load_law(name)
        bh = BellmanHarris(law, lifetime_shape=8.0, lifetime_mean_bounds=[10, 80])
        counts = bh.simulate(bh.physical_from_mapping(values(law, bh)), 144, 300,
                             np.random.default_rng(3))  # fmt: skip
        assert hashlib.sha256(counts.tobytes()).hexdigest()[:16] == digest, name


@pytest.mark.parametrize("fixed", [False, True])
def test_encode_decode_round_trip(fixed):
    het = load_law("two_type_hetero")
    extra = {"fixed": {"density": {"capacity": 250.0}}} if fixed else {}
    bh = make_bh(het, **extra)
    assert ("density.log_capacity" in bh.theta_names) != fixed
    p = bh.physical_from_mapping(values(het, bh, capacity=250.0, kappa=12.0))
    theta = bh.encode(p)
    assert np.allclose(bh.decode(theta), p, rtol=1e-5)
    draws = bh.prior().sample((50,)).numpy()
    phys = bh.decode(draws)
    k = phys[:, bh.parameter_names.index("density.capacity")]
    assert np.all((k >= 1.0) & (k <= 1e13))
    assert np.allclose(np.stack([bh.encode(q) for q in phys]), draws, atol=1e-4)


def test_invalid_configs():
    law = load_law()
    for bad in ({"capacity_bounds": [0, 10]}, {"capacity_bounds": [10, 5]},
                {"capacity_bounds": [1, np.inf]}, {"other": 1}):  # fmt: skip
        with pytest.raises(ValueError):
            make_bh(law, dd=bad)


# ------------------------------------------------------------------ K -> inf
def test_kernel_infinite_capacity_matches_exponential_means():
    # Event-driven kernel with K = inf and exponential lifetimes vs the exact expm means.
    law = load_law()
    bh = make_bh(law, dd=None, shape=1.0)
    p = bh.physical_from_mapping(values(law, bh))
    P, founder, size_probs = bh.law.unpack(bh.space, p)
    counts = bh_density_forward(
        P[None], bh.law.n_outcomes, bh.law.children, bh.law.n_children, founder,
        bh.law.founder_sizes, size_probs, bh.shape, bh.mean_lifetimes(p), np.inf, 100.0,
        20_000, np.random.default_rng(0), bh.max_cells, bh.max_events,
    )  # fmt: skip
    exact = bh.expected_counts(p, 100.0)
    se = counts.std(0) / np.sqrt(len(counts))
    assert np.all(np.abs(counts.mean(0) - exact) < 5 * se)


@pytest.mark.parametrize("name", ["two_type", "two_type_hetero", "multi_founder"])
def test_huge_capacity_reproduces_constant_lifetimes(name):
    law = load_law(name)
    if name == "two_type":  # free founder row: mixed A/B founders
        law["fixed"] = {}
        law["reference"] = {**law["reference"], "founder": {"A": 0.6, "B": 0.4}}
    kappa = 10.0 if name == "two_type_hetero" else None
    dd, off = make_bh(law), make_bh(law, dd=None)
    p = dd.physical_from_mapping(values(law, dd, capacity=1e12, kappa=kappa))
    q = off.physical_from_mapping(values(law, off, kappa=kappa))
    a = dd.simulate(p, 144, 6000, np.random.default_rng(0))
    b = off.simulate(q, 144, 6000, np.random.default_rng(1))
    se = np.sqrt(a.var(0) / len(a) + b.var(0) / len(b))
    assert np.all(np.abs(a.mean(0) - b.mean(0)) < 5 * se)
    assert abs(totals(a)[1] - totals(b)[1]) < 0.05
    assert abs(np.mean(a.sum(1) == 0) - np.mean(b.sum(1) == 0)) < 0.02


# ------------------------------------------------------------------ small K
def test_small_capacity_slows_growth_and_tightens_totals():
    law = load_law()
    small, big = make_bh(law), make_bh(law, dd=None)
    p = small.physical_from_mapping(values(law, small, capacity=30.0))
    q = big.physical_from_mapping(values(law, big))
    out = {}
    for label, sim, par in (("small", small, p), ("big", big, q)):
        t144, sd144 = totals(sim.simulate(par, 144, 3000, np.random.default_rng(0)))
        t240, sd240 = totals(sim.simulate(par, 240, 3000, np.random.default_rng(1)))
        out[label] = (t240.mean(), t240.mean() / t144.mean(), sd240)
    assert out["small"][0] < 0.5 * out["big"][0]
    assert out["small"][1] < 0.75 * out["big"][1]
    assert out["small"][2] < out["big"][2] - 0.1


def test_density_dependence_composes_and_is_reproducible():
    for name in ("two_type_hetero", "multi_founder"):
        law = load_law(name)
        bh = make_bh(law, max_cells=2000)
        kappa = 5.0 if name == "two_type_hetero" else None
        p = bh.physical_from_mapping(values(law, bh, capacity=50.0, kappa=kappa))
        a = bh.simulate(p, 144, 200, np.random.default_rng(9))
        assert np.array_equal(a, bh.simulate(p, 144, 200, np.random.default_rng(9)))
        assert a.min() >= 0


def test_max_cells_marks_invalid():
    law = load_law()
    bh = make_bh(law, max_cells=50)
    p = bh.physical_from_mapping(values(law, bh, capacity=1e12))
    counts = bh.simulate(p, 240, 200, np.random.default_rng(0))
    assert np.any(counts == -1) and np.all((counts == -1).all(1) | (counts >= 0).all(1))


def test_expected_counts_monte_carlo():
    law = load_law()
    bh = make_bh(law, shape=1.0)
    p = bh.physical_from_mapping(values(law, bh, capacity=1e12))
    mc = bh.expected_counts(p, 100.0)
    assert np.array_equal(mc, bh.expected_counts(p, 100.0))  # fixed seed
    exact = make_bh(law, dd=None, shape=1.0).expected_counts(
        bh.physical_from_mapping(values(law, bh, capacity=1e12))[:-1], 100.0
    )
    assert np.allclose(mc, exact, rtol=0.1)
    small = bh.physical_from_mapping(values(law, bh, capacity=20.0))
    assert bh.expected_counts(small, 100.0).sum() < mc.sum()


# ------------------------------------------------------------------ configs
def compose_cfg(*overrides):
    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        return compose("config", overrides=list(overrides))


@pytest.mark.parametrize("law", ["two_type", "two_type_hetero"])
@pytest.mark.parametrize("free_founder", [False, True])
def test_dd_config_composes_and_simulates(law, free_founder):
    extra = ["~law.fixed.founder"] if free_founder else []
    cfg = compose_cfg("simulator=bellman_harris_dd", f"law={law}", "design=real_like",
                      "design.clones=5", "workers=1", *extra)  # fmt: skip
    comp = build_components(cfg)
    data = build_data_generator(cfg, comp.generator)
    truth = reference_parameters(cfg, data.simulator)
    assert truth[data.simulator.parameter_names.index("density.capacity")] == 300.0
    assert "density.log_capacity" in comp.simulator.theta_names
    assert ("clone.log_kappa" in comp.simulator.theta_names) == (law == "two_type_hetero")
    assert data.dataset(truth, "reference") is not None
    theta = comp.simulator.prior().sample((3,)).numpy()
    x, _ = comp.generator.simulate_theta(theta, "t")
    assert x.shape == (3, comp.generator.x_dim)


def test_capacity_can_be_fixed_from_command_line():
    cfg = compose_cfg("simulator=bellman_harris_dd",
                      "+simulator.model.fixed.density.capacity=500")  # fmt: skip
    sim = build_components(cfg).simulator
    assert "density.log_capacity" not in sim.theta_names
    assert sim.space.fixed_scalars["density.capacity"] == 500.0


@pytest.mark.slow
def test_dd_pipeline_smoke_and_real_2d_fit(tmp_path):
    import json

    from omegaconf import OmegaConf

    from bio_sbi.pipeline import infer_measured, run_experiment

    cfg = compose_cfg(
        "simulator=bellman_harris_dd", "law=two_type", "~law.fixed.founder",
        "design=real_like", "budget=smoke", "budget.simulations=128", "budget.max_epochs=2",
        "budget.sbc_cases=4", "budget.posterior_samples=40", "budget.sbc_samples=20",
        "budget.ppc_samples=5", "design.clones=[16,10]", "workers=2",
    )  # fmt: skip
    out = tmp_path / "run"
    (out / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, out / ".hydra" / "config.yaml")
    metrics = run_experiment(cfg, out)
    rows = [r for r in metrics["recovery"]["per_parameter"] if r["parameter"] == "density.capacity"]
    assert rows and rows[0]["inferred"] and rows[0]["truth"] == 300.0
    fit = infer_measured(out, REAL_2D, tmp_path / "fit_2d", n=40, ppc=5)
    measured = json.loads((fit / "metrics.json").read_text())
    assert measured["parameters"]["density.capacity"]["inferred"]
