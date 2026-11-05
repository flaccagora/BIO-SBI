import numpy as np
import pytest

from bio_sbi.simulators import INVALID, BellmanHarris, GaltonWatson

from .conftest import load_law

MULTI = {
    "founder": {"A": 0.6, "B": 0.4},
    "founder_size": {"1": 0.5, "2": 0.3, "3": 0.2},
}


def close(a, b, se, k=5):
    return np.all(np.abs(np.asarray(a) - np.asarray(b)) <= k * se + 1e-9)


def test_gw_mean_matches_theory(gw, gw_truth, rng):
    counts = gw.simulate(gw_truth, 6, 40_000, rng)
    assert close(counts.mean(0), gw.expected_counts(gw_truth, 6), counts.std(0) / 200)


def multi_gw():
    spec = load_law("multi_founder")
    gw = GaltonWatson(spec)
    return gw, gw.physical_from_mapping({**spec["reference"], **MULTI})


def test_multi_founder_mean(rng):
    gw, p = multi_gw()
    counts = gw.simulate(p, 5, 40_000, rng)
    assert close(counts.mean(0), gw.expected_counts(p, 5), counts.std(0) / 200)


def test_single_generation_zero_is_founder_count(rng):
    gw, p = multi_gw()
    # One generation of pure persistence leaves exactly the founders.
    q = p.copy()
    names = gw.parameter_names
    for t in "AB":
        q[gw.space.row_slice(t)] = 0
        q[names.index(f"{t}.persist")] = 1
    counts = gw.simulate(q, 1, 50_000, rng).sum(1)
    assert np.allclose(np.bincount(counts, minlength=4)[1:] / 50_000, [0.5, 0.3, 0.2], atol=0.01)


@pytest.mark.parametrize("multi", [False, True])
def test_gw_detected_sampler_matches_rejection(gw, gw_truth, rng, multi):
    if multi:
        gw, gw_truth = multi_gw()
    capture = np.array([0.6, 0.4])
    exact = gw.simulate_detected(gw_truth, 5, 40_000, capture, rng)
    forward = rng.binomial(gw.simulate(gw_truth, 5, 80_000, rng), capture)
    forward = forward[forward.sum(1) > 0]
    assert np.all(exact.sum(1) > 0)
    se = np.sqrt(exact.var(0) / len(exact) + forward.var(0) / len(forward))
    assert close(exact.mean(0), forward.mean(0), se)
    assert abs(len(forward) / 80_000 - gw.detection_probability(gw_truth, 5, capture)) < 0.01


def test_gw_founder_mixture_reweighted_by_detection():
    # B founders always die, so every detected clone descends only from A founders.
    law = {"types": ["A", "B"], "outcomes": {"A": {"aa": [2, 0]}, "B": {"death": [0, 0]}}}
    gw = GaltonWatson(law)
    p = gw.physical_from_mapping({"founder.A": 0.3, "founder.B": 0.7})
    counts = gw.simulate_detected(p, 3, 200, np.ones(2), np.random.default_rng(0))
    assert np.all(counts == [8, 0])


def test_gw_resource_limit_marks_invalid(law):
    gw = GaltonWatson(law, max_cells=50)
    p = gw.physical_from_mapping({"A.div_AA": 1.0, "B.persist": 1.0})
    assert np.all(gw.simulate(p, 10, 3, np.random.default_rng(0)) == INVALID)


def test_gw_rejects_fractional_horizon(gw, gw_truth, rng):
    with pytest.raises(ValueError):
        gw.simulate(gw_truth, 2.5, 1, rng)


def make_bh(spec, shape=1.0, **kw):
    return BellmanHarris(spec, lifetime_shape=shape, lifetime_mean_bounds=[5, 50], **kw)


@pytest.mark.parametrize("multi", [False, True])
def test_bh_exponential_mean_matches_theory(rng, multi):
    spec = load_law("multi_founder" if multi else "two_type")
    bh = make_bh(spec)
    ref = {**spec["reference"], "A.mean_lifetime": 20, "B.mean_lifetime": 30}
    p = bh.physical_from_mapping({**ref, **(MULTI if multi else {})})
    counts = bh.simulate(p, 120, 40_000, rng)
    assert close(counts.mean(0), bh.expected_counts(p, 120), counts.std(0) / 200)


def test_bh_deterministic_lifetimes_approach_gw(law, gw, gw_truth, rng):
    # Gamma lifetimes with a huge shape are ~deterministic: BH at (G + 0.5) * mean ~ GW at G.
    bh = make_bh(law, shape=1e5)
    bh_counts = bh.simulate(np.r_[gw_truth, 10.0, 10.0], 45.0, 20_000, rng)
    gw_counts = gw.simulate(gw_truth, 4, 20_000, rng)
    se = np.sqrt(bh_counts.var(0) / 20_000 + gw_counts.var(0) / 20_000)
    assert close(bh_counts.mean(0), gw_counts.mean(0), se)


def test_bh_parameters_and_fixed_lifetime(law):
    bh = make_bh(law, fixed={"B": {"mean_lifetime": 30.0}})
    assert bh.parameter_names[-2:] == ("A.mean_lifetime", "B.mean_lifetime")
    assert bh.theta_names[-1] == "A.log_mean_lifetime"  # B lifetime is fixed
    p = bh.physical_from_mapping({**law["reference"], "A.mean_lifetime": 12})
    assert p[-1] == 30.0
    assert np.allclose(bh.decode(bh.encode(p)), p, atol=1e-5)
    assert bh.prior().sample((5,)).shape == (5, bh.theta_dim)


def test_three_type_simulators_run(rng):
    spec = load_law("three_type")
    gw = GaltonWatson(spec)
    p = gw.physical_from_mapping(spec["reference"])
    counts = gw.simulate(p, 6, 5000, rng)
    assert counts.shape == (5000, 3)
    assert close(counts.mean(0), gw.expected_counts(p, 6), counts.std(0) / np.sqrt(5000))
