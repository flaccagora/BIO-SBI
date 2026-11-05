import numpy as np
import pytest

from bio_sbi.design import Design
from bio_sbi.observation import ObservationModel
from bio_sbi.simulate import DataGenerator
from bio_sbi.simulators import GaltonWatson
from bio_sbi.summaries import CohortMeans, CohortMoments

from .conftest import load_law


def make(gw, workers=1, **obs):
    return DataGenerator(
        gw,
        ObservationModel(gw.types, **obs),
        Design([2, 4], [5, 7]),
        CohortMoments(),
        seed=3,
        workers=workers,
    )


def test_design_validation():
    assert Design([1, 2], 4).cohorts == ((1.0, 4), (2.0, 4))
    with pytest.raises(ValueError):
        Design([1, 1], 4)
    with pytest.raises(ValueError):
        Design([1, 2], [4])


def test_dataset_shape_and_selection(gw, gw_truth):
    data = make(gw).dataset(gw_truth, "k")
    assert [c.counts.shape for c in data] == [(5, 2), (7, 2)]
    assert all(np.all(c.counts.sum(1) > 0) for c in data)


@pytest.mark.parametrize("exact", [True, False])
def test_reproducible_across_workers(gw, exact):
    theta = gw.prior().sample((12,)).numpy()
    x1, _ = make(gw, 1, exact_detection=exact).simulate_theta(theta, "s")
    x4, _ = make(gw, 4, exact_detection=exact).simulate_theta(theta, "s")
    assert np.array_equal(np.nan_to_num(x1, nan=-9), np.nan_to_num(x4, nan=-9))
    assert x1.shape == (12, make(gw).x_dim)


def test_invalid_dataset_gives_nan(law):
    gw = GaltonWatson(law, max_cells=10)
    p = gw.physical_from_mapping({"A.div_AA": 1.0, "B.persist": 1.0})
    gen = make(gw)
    assert gen.dataset(p, "x") is None
    assert np.all(np.isnan(gen.summarize(None)))


def test_observation_noise_statistics(rng):
    obs = ObservationModel(["A", "B"], capture=[0.5, 1.0], confusion=[[0.8, 0.2], [0.0, 1.0]])
    measured = obs.measure(np.tile([1000, 0], (200, 1)), rng)
    assert np.allclose(measured.mean(0) / 1000, [0.4, 0.1], atol=0.01)
    assert np.allclose(
        ObservationModel(["A", "B"], confusion=0.1).confusion, [[0.9, 0.1], [0.1, 0.9]]
    )
    with pytest.raises(ValueError):
        ObservationModel(["A", "B"], confusion=[[0.5, 0.4], [0, 1]])


def test_rectangular_confusion_merges_hidden_type(rng):
    obs = ObservationModel(
        ["A", "B", "C"], labels=["A", "B"], confusion={"A": {"A": 1}, "B": {"B": 1}, "C": {"B": 1}}
    )
    out = obs.measure(np.array([[3, 4, 5]]), rng)
    assert out.tolist() == [[3, 9]]
    with pytest.raises(ValueError, match="same-named label"):
        ObservationModel(["A", "B", "C"], labels=["A", "B"])


def test_three_type_data_through_two_markers():
    spec = load_law("three_type")
    gw = GaltonWatson(spec)
    obs = {"labels": ["A", "B"], "confusion": {"A": {"A": 1}, "B": {"B": 1}, "C": {"B": 1}}}
    gen = make(gw, **obs)
    data = gen.dataset(gw.physical_from_mapping(spec["reference"]), "r")
    assert data[0].counts.shape == (5, 2)
    assert gen.x_dim == make(GaltonWatson(load_law())).x_dim


def test_all_clone_selection_keeps_extinct(law):
    gw = GaltonWatson(law)
    ref = {**law["reference"], "A.death": 0.5, "A.div_AA": 0.08}
    gen = DataGenerator(
        gw,
        ObservationModel(gw.types, selection="all", exact_detection=False),
        Design([4], 200),
        CohortMeans(),
    )
    counts = gen.dataset(gw.physical_from_mapping(ref), "z")[0].counts
    assert len(counts) == 200 and np.any(counts.sum(1) == 0)


def test_detection_stats_rejection_vs_exact(gw, gw_truth):
    stats_exact, stats_rej = [], []
    gen_exact = make(gw, capture=0.5)
    gen_rej = make(gw, capture=0.5, exact_detection=False)
    gen_rej.design = Design([4], [4000])
    gen_exact.dataset(gw_truth, "k", stats=stats_exact)
    gen_rej.dataset(gw_truth, "k", stats=stats_rej)
    assert [s["seeded"] for s in stats_exact] == [None, None]
    exact = stats_exact[1]["detection_probability"]
    assert exact == pytest.approx(
        gw.detection_probability(gw_truth, 4, gen_exact.observation.capture)
    )
    (rej,) = stats_rej
    assert rej["recovered"] == 4000 and rej["seeded"] >= 4000
    assert rej["detection_rate"] == pytest.approx(rej["detection_probability"], abs=0.02)
    assert rej["detection_probability"] == pytest.approx(exact)


def test_detection_stats_do_not_change_data(gw, gw_truth):
    gen = make(gw, capture=0.7, exact_detection=False)
    plain = gen.dataset(gw_truth, "k")
    counted = gen.dataset(gw_truth, "k", stats=[])
    assert all(np.array_equal(a.counts, b.counts) for a, b in zip(plain, counted, strict=True))


def test_detection_stats_all_clones(gw, gw_truth):
    stats = []
    make(gw, selection="all", exact_detection=False).dataset(gw_truth, "k", stats=stats)
    assert [s["seeded"] for s in stats] == [5, 7]
    assert all(s["detected"] <= s["seeded"] for s in stats)
