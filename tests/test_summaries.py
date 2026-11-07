"""Summary statistics: SurvivorMoments vs CohortMoments."""

import numpy as np
import pytest

from bio_sbi.summaries import Cohort, CohortMoments, SurvivorMoments


def cohorts(rng, n=40, k=2, empty=0):
    out = []
    for h in (4.0, 8.0):
        counts = rng.poisson(5, size=(n, k)) + np.eye(k, dtype=int)[rng.integers(k, size=n)]
        counts[:empty] = 0
        out.append(Cohort(h, counts))
    return out


@pytest.mark.parametrize("k", [1, 2, 3])
def test_equivalent_without_empty_clones(k):
    data = cohorts(np.random.default_rng(0), k=k)
    assert all((c.counts.sum(1) > 0).all() for c in data)
    np.testing.assert_array_equal(SurvivorMoments()(data), CohortMoments()(data))


def test_survivor_statistics_and_extinct_fraction():
    rng = np.random.default_rng(1)
    data = cohorts(rng, n=40, empty=10)
    labels = ["A", "B"]
    s = SurvivorMoments()
    x = dict(zip(s.feature_names([4, 8], labels), s(data), strict=True))
    survivors = [Cohort(c.horizon, c.counts[10:]) for c in data]
    ref = dict(zip(CohortMoments().feature_names([4, 8], labels), CohortMoments()(survivors),
                   strict=True))  # fmt: skip
    for h in ("h4", "h8"):
        assert x[f"{h}.total.extinct"] == pytest.approx(0.25)
        for name, value in ref.items():
            if name.startswith(h) and not name.endswith("total.empty"):
                assert x[name] == value, name


@pytest.mark.parametrize("n_alive", [0, 1])
def test_no_nan_with_few_survivors(n_alive):
    counts = np.zeros((5, 2), dtype=int)
    counts[:n_alive] = [3, 4]
    x = dict(zip(SurvivorMoments().feature_names([4], ["A", "B"]),
                 SurvivorMoments()([Cohort(4.0, counts)]), strict=True))  # fmt: skip
    assert np.isfinite(list(x.values())).all()
    assert x["h4.total.extinct"] == pytest.approx(1 - n_alive / 5)
    if n_alive == 0:
        assert all(v == 0 for n, v in x.items() if n != "h4.total.extinct")
    else:
        assert x["h4.A.mean_log"] == pytest.approx(np.log1p(3)) and x["h4.A.present"] == 1


def test_feature_names_match_length():
    s = SurvivorMoments(quantiles=(0.5,))
    data = cohorts(np.random.default_rng(2), k=3, empty=3)
    assert len(s.feature_names([4, 8], ["A", "B", "C"])) == len(s(data))
