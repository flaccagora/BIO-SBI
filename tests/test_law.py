import numpy as np
import pytest
import torch

from bio_sbi.simulators import OffspringLaw, ParameterSpace, all_outcomes
from bio_sbi.simulators.law import flatten

from .conftest import load_law


def test_all_outcomes_two_types():
    table = all_outcomes(["A", "B"])
    assert list(table["A"]) == ["death", "persist", "to_B", "div_AA", "div_AB", "div_BB"]
    assert table["B"]["persist"] == [0, 1] and table["B"]["to_A"] == [1, 0]
    assert table["A"]["div_AB"] == [1, 1]
    assert len(all_outcomes(["A", "B", "C"])["C"]) == 1 + 3 + 6


@pytest.mark.parametrize(
    "name",
    ["two_type", "reversible_ab", "hierarchy_ab", "minimal_ab", "three_type", "multi_founder"],
)
def test_shipped_laws_and_references(name):
    spec = load_law(name)
    law = OffspringLaw(**spec)
    space = ParameterSpace(law.rows, fixed=law.fixed, concentration=law.concentration)
    truth = space.complete(spec["reference"])
    assert np.allclose(space.decode(space.encode(truth)), truth, atol=1e-6)


def test_default_law_infers_everything_but_founder():
    spec = load_law()
    space = ParameterSpace(OffspringLaw(**spec).rows, fixed=OffspringLaw(**spec).fixed)
    # 2 types x 6 outcomes -> 10 free; founder fixed to A; one founder size.
    assert space.theta_dim == 10
    assert "founder.A" not in space.free_names


def test_fixed_values_and_leftover_mass():
    law = OffspringLaw(["A", "B"], fixed={"A": {"death": 0.2, "persist": 0.0}, "founder.A": 1})
    space = ParameterSpace(law.rows, fixed=law.fixed)
    theta = space.prior().sample((500,)).numpy()
    p = space.decode(theta)
    names = space.names
    assert np.allclose(p[:, names.index("A.death")], 0.2)
    assert np.allclose(p[:, names.index("A.persist")], 0.0)
    assert np.allclose(p[:, space.row_slice("A")].sum(1), 1.0)
    assert np.allclose(p[:, names.index("founder.B")], 0.0)
    with pytest.raises(ValueError, match="fixed"):
        space.complete({"A.death": 0.3, "A.div_AA": 0.7})


def test_invalid_fixed_values():
    rows = OffspringLaw(["A", "B"]).rows
    with pytest.raises(ValueError):
        ParameterSpace(rows, fixed={"A.death": 0.7, "A.persist": 0.6})
    with pytest.raises(ValueError, match="Unknown"):
        ParameterSpace(rows, fixed={"A.nope": 0.1})


def test_concentration_matches_dirichlet_mean():
    law = OffspringLaw(
        ["A"],
        {"A": {"d": [0], "a": [1], "aa": [2]}},
        concentration={"A": {"d": 2.0, "a": 3.0, "aa": 5.0}},
    )
    space = ParameterSpace(law.rows, concentration=law.concentration)
    torch.manual_seed(0)
    p = space.decode(space.prior().sample((100_000,)).numpy())
    assert np.allclose(p[:, space.row_slice("A")].mean(0), [0.2, 0.3, 0.5], atol=5e-3)


def test_flatten_mixed_forms():
    assert flatten({"A": {"death": 0.1}, "founder_size.2": 0.3, "founder_size": {1: 0.7}}) == {
        "A.death": 0.1,
        "founder_size.2": 0.3,
        "founder_size.1": 0.7,
    }
