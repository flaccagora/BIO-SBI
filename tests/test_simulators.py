import numpy as np
import pytest

from bio_sbi.simulators import INVALID, BellmanHarris, GaltonWatson


def test_gw_mean_matches_theory(gw, gw_truth, rng):
    counts = gw.simulate(gw_truth, 6, 40_000, rng)
    expected = gw.expected_counts(gw_truth, 6)
    se = counts.std(0) / np.sqrt(len(counts))
    assert np.all(np.abs(counts.mean(0) - expected) < 5 * se)


def test_gw_detected_sampler_matches_rejection(gw, gw_truth, rng):
    capture = np.array([0.6, 0.4])
    exact = gw.simulate_detected(gw_truth, 6, 40_000, capture, rng)
    forward = rng.binomial(gw.simulate(gw_truth, 6, 80_000, rng), capture)
    forward = forward[forward.sum(1) > 0]
    assert np.all(exact.sum(1) > 0)
    se = np.sqrt(exact.var(0) / len(exact) + forward.var(0) / len(forward))
    assert np.all(np.abs(exact.mean(0) - forward.mean(0)) < 5 * se)
    p_det = gw.detection_probability(gw_truth, 6, capture)
    assert abs(len(forward) / 80_000 - p_det) < 0.01


def test_gw_founder_mixture_reweighted_by_detection(law):
    # B founders that always die are never detected: all detected clones start from A.
    outcomes = {"A": {"aa": [2, 0]}, "B": {"death": [0, 0]}}
    gw = GaltonWatson(["A", "B"], outcomes, [0.3, 0.7])
    p = gw.physical_from_mapping({"A.aa": 1.0, "B.death": 1.0})
    counts = gw.simulate_detected(p, 3, 100, np.ones(2), np.random.default_rng(0))
    assert np.all(counts == [8, 0])


def test_gw_resource_limit_marks_invalid(law):
    gw = GaltonWatson(law["types"], law["outcomes"], law["founder"], max_cells=50)
    p = gw.physical_from_mapping({"A.self_renew": 1.0, "B.persist": 1.0})
    out = gw.simulate(p, 10, 3, np.random.default_rng(0))
    assert np.all(out == INVALID)


def test_gw_rejects_fractional_horizon(gw, gw_truth, rng):
    with pytest.raises(ValueError):
        gw.simulate(gw_truth, 2.5, 1, rng)


def make_bh(law, shape=1.0):
    return BellmanHarris(
        law["types"],
        law["outcomes"],
        law["founder"],
        lifetime_shape={"A": shape, "B": shape},
        lifetime_mean_bounds={"A": [5, 50], "B": [5, 50]},
    )


def test_bh_exponential_mean_matches_theory(law, rng):
    bh = make_bh(law)
    p = bh.physical_from_mapping({**law["reference"], "A.mean_lifetime": 20, "B.mean_lifetime": 30})
    counts = bh.simulate(p, 120, 40_000, rng)
    se = counts.std(0) / np.sqrt(len(counts))
    assert np.all(np.abs(counts.mean(0) - bh.expected_counts(p, 120)) < 5 * se)


def test_bh_deterministic_lifetimes_approach_gw(law, gw, gw_truth, rng):
    # Gamma lifetimes with huge shape are ~deterministic: BH at T = (G + 0.5) * mean ~ GW at G.
    bh = make_bh(law, shape=1e5)
    p = np.r_[gw_truth, 10.0, 10.0]
    bh_counts = bh.simulate(p, 45.0, 20_000, rng)
    gw_counts = gw.simulate(gw_truth, 4, 20_000, rng)
    se = np.sqrt(bh_counts.var(0) / 20_000 + gw_counts.var(0) / 20_000)
    assert np.all(np.abs(bh_counts.mean(0) - gw_counts.mean(0)) < 5 * se)


def test_bh_parameter_names_and_roundtrip(law):
    bh = make_bh(law)
    assert bh.parameter_names[-2:] == ("A.mean_lifetime", "B.mean_lifetime")
    assert bh.theta_names[-2:] == ("A.log_mean_lifetime", "B.log_mean_lifetime")
    p = bh.physical_from_mapping({**law["reference"], "A.mean_lifetime": 12, "B.mean_lifetime": 40})
    assert np.allclose(bh.decode(bh.encode(p)), p, atol=1e-5)
    sample = bh.prior().sample((5,))
    assert sample.shape == (5, bh.theta_dim)
