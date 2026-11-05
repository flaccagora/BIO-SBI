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
@pytest.mark.parametrize(
    "law",
    ["two_type", "reversible_ab", "hierarchy_ab", "minimal_ab", "three_type", "multi_founder"],
)
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
    mcmc = [] if method == "npe" else ["budget.sbc_cases=2", "budget.ppc_samples=5",
                                       "budget.mcmc_chains=2"]  # fmt: skip
    cfg = config(f"inference={method}", *mcmc)
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
    assert len(runs) == 1 and len(params) == len(metrics["recovery"]["per_parameter"])
    assert metrics["well_specified"] and metrics["ppc"]["features"] > 0


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
    dump = lambda m: json.dumps(m["recovery"]["per_parameter"])  # noqa: E731 (NaN-safe)
    assert dump(a) == dump(b)


@pytest.mark.parametrize("simulator", ["galton_watson", "bellman_harris"])
def test_three_type_data_two_type_model(tmp_path, simulator):
    cfg = config(
        f"simulator={simulator}",
        "law@data.law=three_type",
        "observation@data.observation=three_as_two",
        "budget.sbc_cases=0",
    )
    metrics = run_experiment(cfg, tmp_path)
    assert not metrics["well_specified"] and metrics["data_law"] == "three_type"
    rows = {r["parameter"]: r for r in metrics["recovery"]["per_parameter"]}
    assert rows["A.div_AA"]["truth"] == pytest.approx(0.5)  # matched by name
    assert rows["A.to_B"]["truth"] == 0.0  # exists in both laws, zero in the data
    assert "fraction_outside_95" in metrics["ppc"]


def test_two_type_data_three_type_model(tmp_path):
    cfg = config(
        "law=three_type",
        "observation=three_as_two",
        "law@data.law=two_type",
        "observation@data.observation=perfect",
        "budget.sbc_cases=0",
    )
    metrics = run_experiment(cfg, tmp_path)
    rows = {r["parameter"]: r for r in metrics["recovery"]["per_parameter"]}
    assert np.isnan(rows["A.to_C"]["truth"]) and rows["A.to_C"]["covered"] is None
    assert metrics["recovery"]["scored_parameters"] < metrics["n_free_parameters"] + 3


def test_incompatible_data_models_rejected():
    from bio_sbi.pipeline import build_data_generator

    cfg = config("law@data.law=three_type")  # no label mapping for C
    with pytest.raises(ValueError, match="labels"):
        build_data_generator(cfg, build_components(cfg).generator)
    cfg = config("simulator@data.simulator=bellman_harris")
    with pytest.raises(ValueError, match="time"):
        build_data_generator(cfg, build_components(cfg).generator)


def test_fixed_override_from_command_line():
    cfg = config("+law.fixed.A.persist=0.0", "truth.values={A: {persist: 0.0, div_AA: 0.58}}")
    comp = build_components(cfg)
    assert "A.persist" not in comp.simulator.space.free_names
    assert comp.simulator.theta_dim == 9


# ---------------------------------------------------------------- data seed (P1)
def reference_fingerprint(*overrides):
    import hashlib

    from bio_sbi.pipeline import build_data_generator, reference_parameters

    cfg = config(*overrides)
    data = build_data_generator(cfg, build_components(cfg).generator)
    truth = reference_parameters(cfg, data.simulator)
    digest = hashlib.sha256(np.round(truth, 12).tobytes())
    for cohort in data.dataset(truth, "reference"):
        digest.update(np.ascontiguousarray(cohort.counts, dtype=np.int64).tobytes())
    return digest.hexdigest()[:16]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ((), "81c73a0514ed59c0"),
        (("seed=3",), "d2cd433cb03710ae"),
        (("observation=noisy", "seed=1"), "e8e94e8d3d550f71"),
        (("simulator=bellman_harris",), "6d005ff4a42371e1"),  # rejection sampler
        (("truth.source=prior", "seed=2"), "3b338ccd80349f8f"),
    ],
)
def test_default_data_seed_reproduces_previous_reference(overrides, expected):
    # Digests recorded before data_seed existed (commit 3e85bd8, design.clones=6).
    assert reference_fingerprint(*overrides) == expected


def test_data_seed_controls_reference_only():
    base = reference_fingerprint("truth.source=prior", "seed=0", "data_seed=7")
    assert reference_fingerprint("truth.source=prior", "seed=5", "data_seed=7") == base
    assert reference_fingerprint("truth.source=prior", "seed=0", "data_seed=8") != base
    cfg = config("seed=5", "data_seed=7")
    assert build_components(cfg).generator.seed == 5  # training streams follow `seed`


def test_reference_npz_depends_on_data_seed_only(tmp_path):
    tiny = ["budget.sbc_cases=0", "budget.ppc_samples=0"]

    def reference(name, *overrides):
        metrics = run_experiment(config(*tiny, *overrides), tmp_path / name)
        with np.load(tmp_path / name / "reference.npz") as f:
            return metrics, {k: f[k] for k in f.files}

    m_a, a = reference("a", "seed=0", "data_seed=3")
    m_b, b = reference("b", "seed=1", "data_seed=3")
    _, c = reference("c", "seed=0", "data_seed=4")
    assert a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    assert any(not np.array_equal(a[k], c[k]) for k in a if k.startswith("counts"))
    assert (m_a["seed"], m_a["data_seed"], m_b["seed"]) == (0, 3, 1)
    # Different training seeds still give different posteriors.
    post = lambda m: [r["posterior_mean"] for r in m["recovery"]["per_parameter"]]  # noqa: E731
    assert post(m_a) != post(m_b)


# ---------------------------------------------------------------- detection rate (P2)
def test_reference_detection_in_metrics(tmp_path):
    metrics = run_experiment(
        config("simulator=bellman_harris", "budget.sbc_cases=0", "budget.ppc_samples=0"),
        tmp_path,
    )
    detection = metrics["reference_detection"]
    assert [d["horizon"] for d in detection] == [80.0, 160.0, 240.0]
    for d in detection:
        assert d["recovered"] == 6 and d["seeded"] >= 6
        assert 0 < d["detection_rate"] <= 1 and d["detection_probability"] is None
    runs, _ = collect(tmp_path)
    assert "detection_rate_h80" in runs.columns and runs["data_seed"].iloc[0] == 0


def test_npe_sampling_is_bounded_when_posterior_leaves_the_prior(tmp_path):
    """A wildly out-of-distribution x must not make NPE rejection sampling hang."""
    import time

    from bio_sbi.inference import FittedPosterior

    cfg = config("budget.sbc_cases=0", "budget.ppc_samples=0")
    metrics = run_experiment(cfg, tmp_path)
    assert 0.0 <= metrics["posterior_support_mass"] <= 1.0
    assert metrics["posterior_samples_drawn"] == 40
    fitted = FittedPosterior.load(tmp_path / "posterior.pt")
    fitted.max_sampling_time = 1.0
    x = np.load(tmp_path / "reference.npz")["x"] * 1e4  # far outside the training data
    start = time.perf_counter()
    draws = fitted.sample(x, 100_000, seed=0)
    assert time.perf_counter() - start < 30
    assert len(draws) <= 100_000
    assert 0.0 <= fitted.support_mass(x, n=2000) <= 1.0


def test_npe_sbc_is_bounded_and_counts_leakage(tmp_path):
    """SBC must not hang when the posterior leaks outside the prior for some cases."""
    import time

    from bio_sbi.evaluation import calibration
    from bio_sbi.inference import FittedPosterior

    run_experiment(config("budget.sbc_cases=0", "budget.ppc_samples=0"), tmp_path)
    fitted = FittedPosterior.load(tmp_path / "posterior.pt")
    fitted.max_sampling_time = 0.5
    comp = build_components(config())
    theta = comp.simulator.prior().sample((4,)).numpy()
    x, _ = comp.generator.simulate_theta(theta, "sbc-test")
    x[:2] *= 1e4  # two cases far outside the training data
    start = time.perf_counter()
    metrics, ranks = calibration(fitted, theta, x, comp.simulator.theta_names, 20)
    assert time.perf_counter() - start < 60
    assert metrics["cases"] + metrics["excluded_leakage"] + metrics["invalid_cases"] == 4
    assert ranks.shape == (metrics["cases"], comp.simulator.theta_dim)


def test_data_section_from_runs(tmp_path):
    from bio_sbi.datacard import describe_data, upsert_section, write_data_section

    for seed in (0, 1):
        cfg = config(
            f"seed={seed}", "budget.sbc_cases=0", "budget.ppc_samples=0", "budget.simulations=64"
        )
        out = tmp_path / f"job{seed}"
        save_hydra_config(cfg, out)
        run_experiment(cfg, out)
    text = describe_data(tmp_path)
    assert "Synthetic data" in text and "2 distinct reference datasets" in text
    assert "galton_watson" in text and "A | death 0.03" in text
    (tmp_path / "info.md").write_text("# T\n\n## Question\n\nq\n\n## Results\n\nr\n")
    write_data_section(tmp_path)
    body = (tmp_path / "info.md").read_text()
    assert body.index("## Question") < body.index("## Data") < body.index("## Results")
    write_data_section(tmp_path)  # idempotent: replaced, not duplicated
    assert (tmp_path / "info.md").read_text().splitlines().count("## Data") == 1
    assert upsert_section("# x\n", "## Data\n\nd\n").endswith("## Data\n\nd\n")
