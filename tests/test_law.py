import numpy as np
import pytest
import torch

from bio_sbi.simulators import OffspringLaw

from .conftest import load_law


@pytest.mark.parametrize("name", ["reversible_ab", "hierarchy_ab", "minimal_ab"])
def test_shipped_laws_parse(name):
    spec = load_law(name)
    law = OffspringLaw(spec["types"], spec["outcomes"])
    assert law.n_sticks == law.n_probabilities - len(law.types)
    assert len(law.probability_names) == law.n_probabilities


def test_stick_roundtrip(law):
    law = OffspringLaw(law["types"], law["outcomes"])
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.dirichlet(np.ones(n)) for n in law.n_outcomes])
    assert np.allclose(law.sticks_to_probabilities(law.probabilities_to_sticks(p)), p)


def test_stick_prior_matches_dirichlet():
    law = OffspringLaw(
        ["A", "B"],
        {"A": {"d": [0, 0], "aa": [2, 0], "ab": [1, 1]}, "B": {"d": [0, 0], "b": [0, 1]}},
        concentration={"A": [2.0, 3.0, 5.0], "B": [1.0, 4.0]},
    )
    torch.manual_seed(0)
    sticks = law.stick_prior().sample((200_000,)).numpy()
    p = law.sticks_to_probabilities(sticks)
    expected = np.r_[np.array([2, 3, 5]) / 10, np.array([1, 4]) / 5]
    assert np.allclose(p.mean(0), expected, atol=5e-3)


def test_invalid_law_rejected():
    with pytest.raises(ValueError):
        OffspringLaw(["A"], {"A": {"x": [1, 0]}})
    with pytest.raises(ValueError):
        OffspringLaw(["A", "B"], {"A": {"x": [1, 0]}})


def test_mean_matrix(law, gw, gw_truth):
    m = gw.law.mean_matrix(gw_truth)
    r = law["reference"]
    assert np.isclose(m[0, 0], r["A.persist"] + r["A.asymmetric"] + 2 * r["A.self_renew"])
    assert np.isclose(m[1, 0], r["B.switch"])
