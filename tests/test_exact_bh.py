"""Exact Bellman-Harris likelihood (inference=exact_bh): law, gradients, sampler, pipeline.

Small grids and short horizons so every test runs in seconds.
"""

import json

import numpy as np
import pytest

from bio_sbi.inference.bh_exact import (
    BHModel,
    check_shapes,
    detection_and_grad,
    law_and_jac,
    marginal_laws,
)
from bio_sbi.inference.exact_bh import (
    BHLikelihood,
    ExactBH,
    check_supported,
    kish_ess,
    pareto_k,
)
from bio_sbi.inference.gw_exact import GridSpec
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import BellmanHarris, GaltonWatson
from bio_sbi.summaries import Cohort

from .conftest import load_law

#: Distinct probability for every outcome, so a mislabelled outcome changes the law
#: (the repo's B row is ordered death, to_A, persist, ...).
DISTINCT = {
    "A": {"death": 0.04, "persist": 0.07, "to_B": 0.11, "div_AA": 0.42, "div_AB": 0.15,
          "div_BB": 0.21},
    "B": {"death": 0.13, "persist": 0.05, "to_A": 0.36, "div_AA": 0.19, "div_AB": 0.08,
          "div_BB": 0.19},
    "A.mean_lifetime": 12.0, "B.mean_lifetime": 17.0,
}  # fmt: skip
BOUNDS = [8.0, 40.0]


def bh(law_name="two_type", shape=4.0, **law_overrides):
    return BellmanHarris(load_law(law_name, **law_overrides), lifetime_shape=shape,
                         lifetime_mean_bounds=BOUNDS)  # fmt: skip


def model_for(sim, physical, obs=None, dt=1.0, integrator="dop853"):
    obs = obs or ObservationModel(sim.types)
    P, founder, size_probs = sim.law.unpack(sim.space, physical)
    m = BHModel(sim.law.offspring, sim.law.n_outcomes, check_shapes(sim.shape), obs.capture,
                obs.confusion, founder, sim.law.founder_sizes, size_probs,
                obs.selection == "detected", dt, integrator)  # fmt: skip
    return m, P, sim.mean_lifetimes(physical)


def exact_law(sim, physical, horizons, L=64, obs=None, **kw):
    m, P, means = model_for(sim, physical, obs, **kw)
    return law_and_jac(m, P, means, horizons, GridSpec(L, L, 1.0, 1.0))[0]


def empirical(counts, L):
    h = np.zeros((L, L))
    np.add.at(h, (counts[:, 0], counts[:, 1]), 1)
    return h / len(counts)


def assert_matches(p, counts, n_sim):
    """Every cell of the empirical law within 5 binomial sd (+ slack) of the exact law."""
    L = p.shape[0]
    assert counts.max() < L
    emp = empirical(counts, L)
    sd = np.sqrt(np.clip(p * (1 - p), 0, None) / n_sim)
    assert np.all(np.abs(emp - p) <= 5 * sd + 5 / n_sim)


@pytest.fixture
def distinct():
    sim = bh()
    return sim, sim.physical_from_mapping(DISTINCT)


def test_law_matches_forward_simulation_and_outcome_names(distinct, rng):
    """Exact law vs bh_forward, every outcome with its own probability (B-row order)."""
    sim, phys = distinct
    n_sim = 100_000
    p = exact_law(sim, phys, [20.0, 36.0])
    for i, h in enumerate((20.0, 36.0)):
        assert_matches(p[i], sim.simulate(phys, h, n_sim, rng), n_sim)
    # Swapping B.persist and B.to_A changes the law detectably.
    swapped = {**DISTINCT, "B": {**DISTINCT["B"], "persist": 0.36, "to_A": 0.05}}
    p_swap = exact_law(sim, sim.physical_from_mapping(swapped), [36.0])[0]
    counts = sim.simulate(phys, 36.0, n_sim, rng)
    assert np.abs(empirical(counts, 64) - p_swap).max() > 0.01


def test_normalised_and_one_trajectory_serves_all_horizons(distinct):
    sim, phys = distinct
    p = exact_law(sim, phys, [10.0, 25.0, 40.0], L=128)
    assert p.min() > -1e-14
    np.testing.assert_allclose(p.sum((1, 2)), 1, atol=1e-12)
    single = exact_law(sim, phys, [25.0], L=128)[0]
    np.testing.assert_allclose(single, p[1], atol=1e-14)
    # tails: marginals on a long axis show the table holds the mass
    m, P, means = model_for(sim, phys)
    pa, pb = marginal_laws(m, P, means, [40.0], N=512)
    assert pa[0, 128:].sum() + pb[0, 128:].sum() < 1e-10


@pytest.mark.parametrize("law_name", ["two_type", "reversible_ab"])
def test_shape_one_means_match_expected_counts(law_name):
    sim = bh(law_name, shape=1.0)
    ref = {**load_law(law_name)["reference"], "A.mean_lifetime": 20.0, "B.mean_lifetime": 30.0}
    phys = sim.physical_from_mapping(ref)
    for h in (30.0, 60.0):
        p = exact_law(sim, phys, [h], L=256)[0]
        mean = np.array([p.sum(1) @ np.arange(256), p.sum(0) @ np.arange(256)])
        np.testing.assert_allclose(mean, sim.expected_counts(phys, h), rtol=1e-9)


def test_step_size_convergence(distinct):
    sim, phys = distinct
    p1 = exact_law(sim, phys, [30.0], L=32, dt=1.0)
    p05 = exact_law(sim, phys, [30.0], L=32, dt=0.5)
    p2 = exact_law(sim, phys, [30.0], L=32, dt=2.0)
    assert np.abs(p1 - p05).max() < 1e-12
    assert np.abs(p2 - p05).max() < 1e-10
    rk4 = exact_law(sim, phys, [30.0], L=32, dt=0.25, integrator="rk4")
    assert np.abs(rk4 - p05).max() < 1e-7


def test_per_type_shapes_and_capture_by_pgf_substitution(rng):
    sim = BellmanHarris(load_law(), lifetime_shape={"A": 2, "B": 5}, lifetime_mean_bounds=BOUNDS)
    phys = sim.physical_from_mapping(DISTINCT)
    obs = ObservationModel(sim.types, capture=[0.7, 0.9], confusion=[[0.9, 0.1], [0.2, 0.8]])
    n_sim = 100_000
    counts = obs.measure(sim.simulate(phys, 30.0, n_sim, rng), rng)
    p = exact_law(sim, phys, [30.0], obs=obs)[0]
    assert_matches(p, counts, n_sim)
    m, P, means = model_for(sim, phys, obs)
    q, _ = detection_and_grad(m, P, means, [30.0])
    assert q[0] == pytest.approx(1 - p[0, 0], rel=1e-10)


def test_detection_in_complement_form_and_damped_grid(distinct, rng):
    sim, phys = distinct
    m, P, means = model_for(sim, phys)
    q, _ = detection_and_grad(m, P, means, [15.0, 30.0])
    p = exact_law(sim, phys, [15.0, 30.0], L=128)
    np.testing.assert_allclose(q, 1 - p[:, 0, 0], rtol=1e-11)
    # a nearly-always-dying clone: q is tiny and still accurate (no cancellation)
    dying = {**DISTINCT, "A": {"death": 0.999999, "div_AA": 1e-6},
             "B": {"death": 0.999999, "to_A": 1e-6}}  # fmt: skip
    m2, P2, means2 = model_for(sim, sim.physical_from_mapping(dying))
    q2, _ = detection_and_grad(m2, P2, means2, [150.0])
    # ~ P(founder still alive) = P(Gamma(4, 12 / 4) > 150) ~ 4e-18 (divisions add ~1e-6)
    from scipy.stats import gamma

    assert q2[0] == pytest.approx(gamma.sf(150.0, 4, scale=3.0), rel=1e-4)
    assert q2[0] < 1e-16
    # small damped grid reproduces the large undamped law on the counts it covers
    from bio_sbi.inference.gw_exact import grid_spec

    counts = sim.simulate(phys, 30.0, 2000, rng)
    g = grid_spec(int(counts[:, 0].max()), int(counts[:, 1].max()))
    pd_, _ = law_and_jac(m, P, means, [30.0], g)
    ma, mb = counts.max(0)  # the grid is designed for counts up to the largest observed
    a, b = np.nonzero(p[1, : ma + 1, : mb + 1] > 1e-12)
    np.testing.assert_allclose(pd_[0, a, b], p[1, a, b], rtol=1e-6)


def detected(sim, phys, h, n, rng):
    c = sim.simulate(phys, h, 4 * n, rng)
    return c[c.sum(1) > 0][:n]


@pytest.mark.parametrize("law_name", ["two_type", "reversible_ab"])
def test_gradient_matches_finite_differences(law_name, rng):
    sim = bh(law_name)
    ref = {**load_law(law_name)["reference"], "A.mean_lifetime": 12.0, "B.mean_lifetime": 18.0}
    truth = sim.physical_from_mapping(ref)
    obs = ObservationModel(sim.types)
    cohorts = [Cohort(h, detected(sim, truth, h, 40, rng)) for h in (20.0, 40.0)]
    lik = BHLikelihood(sim, obs, cohorts)
    mixed = 0.8 * truth + 0.2 * sim.decode(np.full(sim.theta_dim, 0.5))
    mixed[-2:] = [12.0, 18.0]
    theta = sim.encode(mixed).astype(float)
    ll, g, bhhh = lik.evaluate(theta, grad=True)
    assert np.isfinite(ll) and g.shape == (sim.theta_dim,)
    assert np.all(np.linalg.eigvalsh(bhhh) > -1e-8)
    eps = 1e-5
    for i in range(sim.theta_dim):
        e = np.zeros_like(theta)
        e[i] = eps
        fd = (lik.evaluate(theta + e)[0] - lik.evaluate(theta - e)[0]) / (2 * eps)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-5 * np.abs(g).max())
    # u coordinates: gradient consistent too
    u = lik.u_from_theta(theta)
    _, gu, _ = lik.log_posterior_u(u, grad=True)
    for i in (0, sim.theta_dim - 1):  # a stick and a log lifetime
        e = np.zeros_like(u)
        e[i] = eps
        fd = (lik.log_posterior_u(u + e) - lik.log_posterior_u(u - e)) / (2 * eps)
        assert gu[i] == pytest.approx(fd, rel=1e-5, abs=1e-5 * np.abs(gu).max())
    lik.close()


def test_importance_sampling_helpers():
    rng = np.random.default_rng(0)
    assert kish_ess(np.zeros(50)) == pytest.approx(50)
    assert pareto_k(rng.normal(size=1000) * 0.1) < 0.5
    assert pareto_k(np.log(rng.pareto(1.0, 4000) + 1)) > 0.7


def test_sampler_recovers_a_toy_posterior(rng):
    """Two free parameters (A.div_AA vs death, A lifetime): AMIS vs a brute-force grid."""
    law = load_law("reversible_ab")
    law["fixed"] = {
        "founder": {"A": 1.0},
        "A": {"persist": 0.05, "to_B": 0.05, "div_AB": 0.05, "div_BB": 0.25},
        "B": {"death": 0.1, "persist": 0.1, "to_A": 0.8, "div_AA": 0, "div_AB": 0, "div_BB": 0},
    }
    sim = BellmanHarris(law, lifetime_shape=3.0, lifetime_mean_bounds=BOUNDS,
                        fixed={"B.mean_lifetime": 15.0})  # fmt: skip
    truth = sim.physical_from_mapping({"A": {"death": 0.1, "div_AA": 0.5},
                                       "A.mean_lifetime": 12.0})  # fmt: skip
    obs = ObservationModel(sim.types)
    cohorts = [Cohort(h, detected(sim, truth, h, 60, rng)) for h in (24.0, 48.0)]
    method = ExactBH(threads=2, particles=150, max_iterations=4, target_ess=300,
                     init_candidates=4, seed=1)  # fmt: skip
    lik = method.likelihood(sim, obs, cohorts)
    theta, w, diag = method.run(lik)
    assert sim.theta_dim == 2 and diag["ess"] > 100 and diag["pareto_k"] < 1.0
    # brute force on a u-grid covering +-7 sd of the particles
    u = lik.u_from_theta(theta)
    mu = w @ u
    sd = np.sqrt(w @ (u - mu) ** 2)
    axes = [np.linspace(mu[j] - 7 * sd[j], mu[j] + 7 * sd[j], 41) for j in range(2)]
    U = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    lp = lik.log_posterior_batch(U)
    pw = np.exp(lp - lp.max())
    pw /= pw.sum()
    T = lik.theta_from_u(U)
    for j in range(2):
        m_ref = pw @ T[:, j]
        sd_ref = np.sqrt(pw @ (T[:, j] - m_ref) ** 2)
        m_is = w @ theta[:, j]
        assert abs(m_is - m_ref) < 0.15 * sd_ref + 1e-3
        assert np.sqrt(w @ (theta[:, j] - m_is) ** 2) == pytest.approx(sd_ref, rel=0.2)
    lik.close()


def test_unsupported_configurations_raise():
    obs = ObservationModel(("A", "B"))
    with pytest.raises(ValueError, match="bellman_harris"):
        check_supported(GaltonWatson(load_law()), obs)
    dd = BellmanHarris(load_law(), lifetime_shape=4.0, lifetime_mean_bounds=BOUNDS,
                       density_dependence=True)  # fmt: skip
    with pytest.raises(ValueError, match="density"):
        check_supported(dd, obs)
    with pytest.raises(ValueError, match="heterogeneity"):
        check_supported(bh(clone_heterogeneity={"kappa_bounds": [1, 100]}), obs)
    three = BellmanHarris(load_law("three_type"), lifetime_shape=4.0, lifetime_mean_bounds=BOUNDS)
    with pytest.raises(ValueError, match="two cell types"):
        check_supported(three, ObservationModel(three.types))
    with pytest.raises(ValueError, match="founder"):
        check_supported(bh(fixed={}), obs)
    with pytest.raises(NotImplementedError, match="integral equation"):
        check_supported(bh(shape=2.5), obs)


def test_pipeline_run_writes_standard_artifacts(tmp_path):
    from bio_sbi.collect import collect
    from bio_sbi.pipeline import build_components, run_experiment

    from .test_pipeline import config, save_hydra_config

    cfg = config(
        "inference=exact_bh", "simulator=bellman_harris", "law=reversible_ab",
        "simulator.model.lifetime_shape=2", "simulator.model.lifetime_mean_bounds=[8,40]",
        "data.simulator.model.lifetime_shape=2", "data.simulator.reference.A.mean_lifetime=12",
        "data.simulator.reference.B.mean_lifetime=18",
        "design.clones=12", "design.horizons=[20,36]",
        "inference.method.particles=24", "inference.method.max_iterations=2",
        "inference.method.init_candidates=2", "inference.method.opt_maxiter=4",
        "inference.sbc_cases=2", "inference.method.sbc.particles=16",
        "inference.method.sbc.max_iterations=1", "inference.method.sbc.opt_maxiter=2",
        "inference.method.sbc.init_candidates=1",
        "budget.ppc_samples=5", "budget.sbc_samples=10",
    )  # fmt: skip
    out = tmp_path / "exact"
    save_hydra_config(cfg, out)
    metrics = run_experiment(cfg, out)
    for name in ("posterior.pt", "metrics.json", "recovery.csv", "posterior_samples.csv.gz",
                 "reference.npz", "ppc.npz", "sbc.npz", "figures/posterior.png"):  # fmt: skip
        assert (out / name).exists(), name
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["inference"] == "exact_bh" and saved["posterior_support_mass"] == 1.0
    assert saved["posterior_samples_drawn"] == 40
    d = saved["exact_bh"]
    assert d["sampler"] == "amis" and d["particles"] == 48 and "A.mean_lifetime" in d["weighted_sd"]
    assert d["grid_check"][0]["max_rel_error"] < 1e-5
    cal = saved["calibration"]
    assert cal["cases"] + cal["invalid_cases"] + cal["excluded_leakage"] == 2
    assert metrics["ppc"]["features"] > 0
    runs, _ = collect(tmp_path)
    assert len(runs) == 1 and runs["inference"][0] == "exact_bh"
    assert build_components(cfg).method.exact
    # Unsupported inference model fails with a clear message.
    bad = config("inference=exact_bh", "budget.sbc_cases=0")
    with pytest.raises(ValueError, match="bellman_harris"):
        run_experiment(bad, tmp_path / "bad")
