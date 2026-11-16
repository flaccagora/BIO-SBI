"""Derived quantities (growth rate per hour, stable B:A ratio) for GW and BH draws."""

import numpy as np
import pytest

from bio_sbi.derived import derived_quantities, mean_matrices
from bio_sbi.simulators import BellmanHarris, GaltonWatson

from .conftest import load_law

REF = load_law("reversible_ab")["reference"]


def test_gw_growth_and_ratio_match_expected_counts():
    sim = GaltonWatson(load_law("reversible_ab"), hours_per_generation=24.0)
    phys = sim.physical_from_mapping(REF)
    d = derived_quantities(sim, phys[None]).iloc[0]
    m40, m41 = sim.expected_counts(phys, 40 * 24.0), sim.expected_counts(phys, 41 * 24.0)
    assert d.growth_rate_per_h == pytest.approx(np.log(m41.sum() / m40.sum()) / 24.0, rel=1e-6)
    assert d.stable_B_to_A == pytest.approx(m41[1] / m41[0], rel=1e-6)
    assert d.A_self_renewal == pytest.approx(0.55)
    assert d.doubling_time_h == pytest.approx(np.log(2) / d.growth_rate_per_h)
    M = mean_matrices(sim, phys[None])[0]
    np.testing.assert_allclose(M, sim.law.mean_matrix(sim.law.unpack(sim.space, phys)[0]))


def test_bh_markov_case_matches_matrix_exponential_means():
    """shape 1: means are a matrix exponential (closed form in the simulator)."""
    sim = BellmanHarris(load_law("reversible_ab"), lifetime_shape=1.0, lifetime_mean_bounds=[10.0, 80.0])
    phys = sim.physical_from_mapping({**REF, "A.mean_lifetime": 25.0, "B.mean_lifetime": 30.0})
    d = derived_quantities(sim, phys[None]).iloc[0]
    m1, m2 = sim.expected_counts(phys, 2000.0), sim.expected_counts(phys, 2100.0)
    assert d.growth_rate_per_h == pytest.approx(np.log(m2.sum() / m1.sum()) / 100.0, rel=1e-5)
    assert d.stable_B_to_A == pytest.approx(m2[1] / m2[0], rel=1e-5)


def test_bh_tends_to_gw_for_synchronous_equal_lifetimes():
    phys_map = {**REF, "A.mean_lifetime": 25.0, "B.mean_lifetime": 25.0}
    bh = BellmanHarris(load_law("reversible_ab"), lifetime_shape=200.0, lifetime_mean_bounds=[10.0, 80.0])
    gw = GaltonWatson(load_law("reversible_ab"), hours_per_generation=25.0)
    d_bh = derived_quantities(bh, bh.physical_from_mapping(phys_map)[None]).iloc[0]
    d_gw = derived_quantities(gw, gw.physical_from_mapping(REF)[None]).iloc[0]
    assert d_bh.growth_rate_per_h == pytest.approx(d_gw.growth_rate_per_h, rel=1e-2)
    assert d_bh.stable_B_to_A == pytest.approx(d_gw.stable_B_to_A, rel=1e-2)
    # slower B (longer lifetime) -> more living B per A at the same outcome probabilities
    slow = derived_quantities(bh, bh.physical_from_mapping(
        {**phys_map, "B.mean_lifetime": 30.0})[None]).iloc[0]  # fmt: skip
    assert slow.stable_B_to_A > d_bh.stable_B_to_A
