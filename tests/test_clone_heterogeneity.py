"""Organoid-level heterogeneity (``law.clone_heterogeneity``, random-effects law)."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_module

from bio_sbi.pipeline import build_components, build_data_generator, reference_parameters
from bio_sbi.simulators import BellmanHarris, GaltonWatson, OffspringLaw
from bio_sbi.summaries import Cohort, CohortMoments

from .conftest import load_law

HETERO = {"kappa_bounds": [1.0, 1000.0]}
DIVISIONS = ["div_AA", "div_AB", "div_BB"]
REAL_2D = Path(__file__).resolve().parents[1] / "data" / "real" / "crc0322_2d_bh.csv"


def hetero_law(name="two_type", **hetero):
    return load_law(name, clone_heterogeneity={**HETERO, **hetero})


def make_bh(spec, **kw):
    return BellmanHarris(spec, lifetime_shape=8.0, lifetime_mean_bounds=[10, 80], **kw)


def bh_values(spec, kappa=None):
    values = {**spec["reference"], "A.mean_lifetime": 20.0, "B.mean_lifetime": 30.0}
    if kappa is not None:
        values["clone.kappa"] = kappa
    return values


def stats(counts, horizon=6.0):
    """corr_A_B, total.sd_log and the sd of the A fraction across clones."""
    summary = CohortMoments()
    x = dict(zip(summary.feature_names([horizon], ["A", "B"]),
                 summary([Cohort(horizon, counts)]), strict=True))  # fmt: skip
    total = counts.sum(1)
    frac = counts[total > 0, 0] / total[total > 0]
    return x[f"h{horizon:g}.corr_A_B"], x[f"h{horizon:g}.total.sd_log"], frac.std()


# ------------------------------------------------------------------ parameters
def test_off_by_default_and_kappa_is_last_scalar():
    assert not OffspringLaw(**load_law()).heterogeneous
    assert not OffspringLaw(**load_law(clone_heterogeneity=None)).heterogeneous
    gw = GaltonWatson(hetero_law())
    assert gw.heterogeneous
    assert gw.parameter_names[-1] == "clone.kappa" and gw.theta_names[-1] == "clone.log_kappa"
    bh = make_bh(hetero_law())
    assert bh.parameter_names[-3:] == ("A.mean_lifetime", "B.mean_lifetime", "clone.kappa")
    assert bh.theta_names[-1] == "clone.log_kappa"
    p = bh.physical_from_mapping(bh_values(hetero_law(), kappa=7.0))
    assert np.allclose(bh.scalar_values(p), [20.0, 30.0]) and bh.clone_kappa(p) == 7.0


@pytest.mark.parametrize("fixed_kappa", [False, True])
def test_encode_decode_round_trip(fixed_kappa):
    spec = hetero_law("reversible_ab")
    fixed = {"clone": {"kappa": 30.0}} if fixed_kappa else None
    for sim in (GaltonWatson(spec, fixed=fixed), make_bh(spec, fixed=fixed)):
        values = bh_values(spec) if isinstance(sim, BellmanHarris) else dict(spec["reference"])
        values["clone.kappa"] = 30.0
        p = sim.physical_from_mapping(values)
        assert np.allclose(sim.decode(sim.encode(p)), p, atol=1e-4)
        assert ("clone.log_kappa" in sim.theta_names) != fixed_kappa
        theta = sim.prior().sample((2000,)).numpy()
        kappa = sim.decode(theta)[:, -1]
        assert np.all((kappa >= 1.0 - 1e-4) & (kappa <= 1000.0 + 1e-2))
        if not fixed_kappa:  # log-uniform prior: median ~ sqrt(1 * 1000)
            assert 20 < np.median(kappa) < 50


def test_invalid_configs():
    with pytest.raises(ValueError, match="kappa_bounds"):
        OffspringLaw(**hetero_law(kappa_bounds=[0, 10]))
    with pytest.raises(ValueError, match="Unknown clone_heterogeneity"):
        OffspringLaw(**load_law(clone_heterogeneity={"kapa_bounds": [1, 2]}))
    with pytest.raises(ValueError, match="has no"):
        OffspringLaw(**hetero_law(outcomes=["div_XY"]))


# ------------------------------------------------------------------ clone matrices
@pytest.mark.parametrize("outcomes", [None, DIVISIONS])
def test_clone_matrices_moments_and_fixed_entries(outcomes):
    spec = hetero_law("reversible_ab", outcomes=outcomes)
    spec["fixed"] = {**spec["fixed"], "A": {"persist": 0.03}}
    gw = GaltonWatson(spec)
    kappa = 5.0
    p = gw.physical_from_mapping({**spec["reference"], "clone.kappa": kappa})
    P, _, _ = gw.law.unpack(gw.space, p)
    Pc = gw.clone_matrices(p, P, 40_000, np.random.default_rng(0))
    assert Pc.shape == (40_000, *P.shape)
    assert np.allclose(Pc.sum(-1), 1.0)
    assert np.all(Pc[:, P == 0] == 0)  # zeros (incl. fixed B divisions) stay zero
    persist = gw.law.outcomes["A"]
    assert np.all(Pc[:, 0, list(persist).index("persist")] == 0.03)  # fixed entry kept
    assert np.allclose(Pc.mean(0), P, atol=0.01)
    a = list(persist).index("div_AA")
    # Var = p (m - p) / (kappa + 1), with m the mass of the varying entries of the row.
    varying = [list(persist).index(n) for n in (outcomes or persist) if n != "persist"]
    m = P[0, varying].sum()
    assert Pc[:, 0, a].var() == pytest.approx(P[0, a] * (m - P[0, a]) / (kappa + 1), rel=0.05)
    if outcomes:  # the non-division entries are unchanged in every clone
        assert np.all(Pc[:, 0, list(persist).index("death")] == P[0, 0])
        assert np.allclose(Pc[:, 0, varying].sum(-1), m)


def test_tiny_kappa_limit_puts_all_mass_on_one_entry():
    spec = hetero_law(kappa_bounds=[1e-6, 10])
    gw = GaltonWatson(spec)
    p = gw.physical_from_mapping({**spec["reference"], "clone.kappa": 1e-6})
    P, _, _ = gw.law.unpack(gw.space, p)
    Pc = gw.clone_matrices(p, P, 5000, np.random.default_rng(1))
    assert np.allclose(Pc.sum(-1), 1.0) and np.allclose(Pc.max(-1), 1.0)
    assert np.allclose(Pc.mean(0), P, atol=0.03)  # still E[p_clone] = p


# ------------------------------------------------------------------ kappa -> inf
def test_large_kappa_reproduces_homogeneous_gw(gw, gw_truth):
    spec = hetero_law()
    hetero = GaltonWatson(spec)
    p = hetero.physical_from_mapping({**spec["reference"], "clone.kappa": 1e7})
    a = hetero.simulate(p, 6, 20_000, np.random.default_rng(0))
    b = gw.simulate(gw_truth, 6, 20_000, np.random.default_rng(1))
    se = np.sqrt(a.var(0) / len(a) + b.var(0) / len(b))
    assert np.all(np.abs(a.mean(0) - b.mean(0)) < 5 * se)
    assert np.allclose(stats(a), stats(b), atol=0.03)


def test_large_kappa_reproduces_homogeneous_bh(law):
    spec = hetero_law()
    hetero, homo = make_bh(spec), make_bh(law)
    p = hetero.physical_from_mapping(bh_values(spec, kappa=1e7))
    q = homo.physical_from_mapping(bh_values(law))
    a = hetero.simulate(p, 144, 10_000, np.random.default_rng(0))
    b = homo.simulate(q, 144, 10_000, np.random.default_rng(1))
    se = np.sqrt(a.var(0) / len(a) + b.var(0) / len(b))
    assert np.all(np.abs(a.mean(0) - b.mean(0)) < 5 * se)
    assert np.allclose(stats(a, 144), stats(b, 144), atol=0.04)


# ------------------------------------------------------------------ small kappa
@pytest.mark.parametrize("outcomes", [None, DIVISIONS])
def test_small_kappa_lowers_correlation_and_spreads_composition(gw, gw_truth, outcomes):
    spec = hetero_law(outcomes=outcomes)
    hetero = GaltonWatson(spec)
    p = hetero.physical_from_mapping({**spec["reference"], "clone.kappa": 2.0})
    capture = np.ones(2)
    a = hetero.simulate_detected(p, 6, 4000, capture, np.random.default_rng(0))
    b = gw.simulate_detected(gw_truth, 6, 4000, capture, np.random.default_rng(1))
    corr_a, _, frac_a = stats(a)
    corr_b, _, frac_b = stats(b)
    assert corr_a < corr_b - 0.2
    assert frac_a > frac_b + 0.03


def test_small_kappa_bh_negative_correlation():
    spec = hetero_law(outcomes=DIVISIONS)
    bh = make_bh(spec)
    p = bh.physical_from_mapping(bh_values(spec, kappa=1.0))
    counts = bh.simulate(p, 144, 4000, np.random.default_rng(0))
    counts = counts[counts.sum(1) > 0]
    corr, _, frac = stats(counts, 144)
    homo = make_bh(load_law())
    ref = homo.simulate(homo.physical_from_mapping(bh_values(load_law())), 144, 4000,
                        np.random.default_rng(1))  # fmt: skip
    corr_ref, _, frac_ref = stats(ref[ref.sum(1) > 0], 144)
    assert corr < corr_ref - 0.2 and frac > frac_ref


# ------------------------------------------------------------------ detection
@pytest.mark.parametrize("outcomes", [None, DIVISIONS])
def test_gw_detected_hetero_matches_rejection(outcomes):
    spec = load_law("multi_founder", clone_heterogeneity={**HETERO, "outcomes": outcomes})
    gw = GaltonWatson(spec)
    p = gw.physical_from_mapping({**spec["reference"], "clone.kappa": 3.0})
    capture = np.array([0.6, 0.4])
    rng = np.random.default_rng(3)
    exact = gw.simulate_detected(p, 5, 30_000, capture, rng)
    forward = rng.binomial(gw.simulate(p, 5, 60_000, rng), capture)
    forward = forward[forward.sum(1) > 0]
    assert np.all(exact.sum(1) > 0)
    se = np.sqrt(exact.var(0) / len(exact) + forward.var(0) / len(forward))
    assert np.all(np.abs(exact.mean(0) - forward.mean(0)) < 5 * se)
    assert abs(stats(exact, 5)[0] - stats(forward, 5)[0]) < 0.03
    assert abs(len(forward) / 60_000 - gw.detection_probability(p, 5, capture)) < 0.01


def test_gw_detected_hetero_budget_marks_invalid():
    # Detection is (almost) impossible: capture 1e-6 -> the parameter-rejection budget runs out.
    spec = hetero_law()
    gw = GaltonWatson(spec, max_detection_draws=5)
    p = gw.physical_from_mapping({**spec["reference"], "clone.kappa": 3.0})
    out = gw.simulate_detected(p, 2, 10, np.full(2, 1e-6), np.random.default_rng(0))
    assert np.all(out == -1)


def test_hetero_is_reproducible():
    spec = hetero_law()
    for sim in (GaltonWatson(spec), make_bh(spec)):
        values = bh_values(spec, 5.0) if isinstance(sim, BellmanHarris) else {
            **spec["reference"], "clone.kappa": 5.0}  # fmt: skip
        p = sim.physical_from_mapping(values)
        h = 144 if isinstance(sim, BellmanHarris) else 6
        a = sim.simulate(p, h, 50, np.random.default_rng(9))
        b = sim.simulate(p, h, 50, np.random.default_rng(9))
        assert np.array_equal(a, b)


# ------------------------------------------------------------------ configs
def compose_cfg(*overrides):
    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        return compose("config", overrides=list(overrides))


@pytest.mark.parametrize("simulator", ["galton_watson", "bellman_harris"])
@pytest.mark.parametrize("law", ["two_type_hetero", "reversible_ab_hetero"])
@pytest.mark.parametrize("free_founder", [False, True])
def test_hetero_laws_compose_and_simulate(simulator, law, free_founder):
    extra = ["~law.fixed.founder"] if free_founder else []
    cfg = compose_cfg(f"simulator={simulator}", f"law={law}", "design=real_like",
                      "design.clones=5", "workers=1", *extra)  # fmt: skip
    comp = build_components(cfg)
    data = build_data_generator(cfg, comp.generator)
    truth = reference_parameters(cfg, data.simulator)
    names = data.simulator.parameter_names
    assert truth[names.index("clone.kappa")] == 10.0
    assert "clone.log_kappa" in comp.simulator.theta_names
    assert ("founder.stick_A" in comp.simulator.theta_names) == free_founder
    assert data.dataset(truth, "reference") is not None
    theta = comp.simulator.prior().sample((3,)).numpy()
    x, _ = comp.generator.simulate_theta(theta, "t")
    assert x.shape == (3, comp.generator.x_dim)


@pytest.mark.slow
def test_hetero_pipeline_smoke_and_real_2d_fit(tmp_path):
    import json

    from omegaconf import OmegaConf

    from bio_sbi.pipeline import infer_measured, run_experiment

    cfg = compose_cfg(
        "simulator=bellman_harris", "law=two_type_hetero", "~law.fixed.founder",
        "design=real_like", "budget=smoke", "budget.simulations=128", "budget.max_epochs=2",
        "budget.sbc_cases=4", "budget.posterior_samples=40", "budget.sbc_samples=20",
        "budget.ppc_samples=5", "design.clones=[16,10]", "workers=2",
    )  # fmt: skip
    out = tmp_path / "run"
    (out / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, out / ".hydra" / "config.yaml")
    metrics = run_experiment(cfg, out)
    kappa = [r for r in metrics["recovery"]["per_parameter"] if r["parameter"] == "clone.kappa"]
    assert kappa and kappa[0]["inferred"] and kappa[0]["truth"] == 10.0
    assert metrics["recovery"]["mean_rel_error_scalars"] is not None
    fit = infer_measured(out, REAL_2D, tmp_path / "fit_2d", n=40, ppc=5)
    measured = json.loads((fit / "metrics.json").read_text())
    assert "clone.kappa" in measured["parameters"]
    assert measured["parameters"]["clone.kappa"]["inferred"]
