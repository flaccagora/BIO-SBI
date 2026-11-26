"""Exact Galton-Watson likelihood (inference=exact_gw): law, gradients, NUTS, pipeline."""

import json

import numpy as np
import pytest

from bio_sbi.inference.exact_gw import ExactLikelihood, check_supported
from bio_sbi.inference.gw_exact import (
    GridSpec,
    GWModel,
    detection_and_grad,
    grid_spec,
    law_and_jac,
)
from bio_sbi.inference.nuts import ess, run_chain, split_rhat
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import BellmanHarris, GaltonWatson
from bio_sbi.summaries import Cohort

from .conftest import load_law

#: Distinct probability for every outcome, so a mislabelled outcome changes the law.
DISTINCT = {
    "A": {"death": 0.04, "persist": 0.07, "to_B": 0.11, "div_AA": 0.42, "div_AB": 0.15,
          "div_BB": 0.21},
    "B": {"death": 0.13, "persist": 0.05, "to_A": 0.36, "div_AA": 0.19, "div_AB": 0.08,
          "div_BB": 0.19},
}  # fmt: skip


def model_for(sim, physical, obs=None):
    obs = obs or ObservationModel(sim.types)
    P, founder, size_probs = sim.law.unpack(sim.space, physical)
    m = GWModel(sim.law.offspring, sim.law.n_outcomes, obs.capture, obs.confusion, founder,
                sim.law.founder_sizes, size_probs, obs.selection == "detected")  # fmt: skip
    return m, P


def exact_law(sim, physical, n, L=64, obs=None):
    m, P = model_for(sim, physical, obs)
    p, _ = law_and_jac(m, P, np.zeros((0, P.shape[1])), np.zeros(0, np.int64), n,
                       GridSpec(L, L, 1.0, 1.0))  # fmt: skip
    return p


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
def distinct_gw():
    sim = GaltonWatson(load_law())
    return sim, sim.physical_from_mapping(DISTINCT)


def test_law_is_normalised_and_has_exact_means(distinct_gw):
    sim, phys = distinct_gw
    for n in (1, 3, 5):
        p = exact_law(sim, phys, n, L=128)
        assert p.min() > -1e-14 and p.sum() == pytest.approx(1, abs=1e-12)
        mean = np.array([p.sum(1) @ np.arange(128), p.sum(0) @ np.arange(128)])
        np.testing.assert_allclose(mean, sim.expected_counts(phys, n), rtol=1e-10)


def test_outcome_mapping_matches_forward_simulation(distinct_gw, rng):
    """Law from the pgf vs gw_forward with a different probability for every outcome."""
    sim, phys = distinct_gw
    n_sim = 200_000
    for n in (1, 3):
        counts = sim.simulate(phys, n, n_sim, rng)
        assert_matches(exact_law(sim, phys, n), counts, n_sim)
    # Swapping two B outcomes (persist <-> to_A) changes the law detectably.
    swapped = {**DISTINCT, "B": {**DISTINCT["B"], "persist": 0.36, "to_A": 0.05}}
    p_swap = exact_law(sim, sim.physical_from_mapping(swapped), 2)
    counts = sim.simulate(phys, 2, n_sim, rng)
    assert np.abs(empirical(counts, 64) - p_swap).max() > 0.02


def test_detection_conditioning_and_damping(distinct_gw, rng):
    sim, phys = distinct_gw
    n, n_sim = 4, 100_000
    m, P = model_for(sim, phys)
    D0, R0 = np.zeros((0, P.shape[1])), np.zeros(0, np.int64)
    q, _ = detection_and_grad(m, P, D0, R0, n)
    assert q == pytest.approx(sim.detection_probability(phys, n, np.ones(2)), rel=1e-12)
    p = exact_law(sim, phys, n, L=128)
    assert 1 - p[0, 0] == pytest.approx(q, rel=1e-10)
    cond = p / q
    cond[0, 0] = 0
    counts = sim.simulate_detected(phys, n, n_sim, np.ones(2), rng)
    assert counts.sum(1).min() > 0
    assert_matches(cond, counts, n_sim)
    # A small damped grid reproduces the large undamped one on the counts it covers.
    g = grid_spec(int(counts[:, 0].max()), int(counts[:, 1].max()))
    pd_, _ = law_and_jac(m, P, D0, R0, n, g)
    a, b = np.nonzero(p[: g.Lx, : g.Ly] > 1e-12)
    np.testing.assert_allclose(pd_[a, b], p[a, b], rtol=1e-6)


def test_capture_and_confusion_by_pgf_substitution(distinct_gw, rng):
    sim, phys = distinct_gw
    obs = ObservationModel(sim.types, capture=[0.7, 0.9], confusion=[[0.9, 0.1], [0.2, 0.8]])
    n, n_sim = 3, 200_000
    counts = obs.measure(sim.simulate(phys, n, n_sim, rng), rng)
    p = exact_law(sim, phys, n, obs=obs)
    assert_matches(p, counts, n_sim)
    m, P = model_for(sim, phys, obs)
    q, _ = detection_and_grad(m, P, np.zeros((0, 6)), np.zeros(0, np.int64), n)
    assert q == pytest.approx(1 - p[0, 0], rel=1e-10)
    assert q == pytest.approx(sim.detection_probability(phys, n, obs.capture), rel=1e-12)


@pytest.mark.parametrize("kernel", ["loop", "adjoint"])
@pytest.mark.parametrize("law_name", ["two_type", "reversible_ab"])
def test_gradient_matches_finite_differences(law_name, kernel, rng):
    sim = GaltonWatson(load_law(law_name))
    truth = sim.physical_from_mapping(load_law(law_name)["reference"])
    obs = ObservationModel(sim.types)
    cohorts = [Cohort(h, sim.simulate_detected(truth, h, 64, np.ones(2), rng)) for h in (3, 6)]
    lik = ExactLikelihood(sim, obs, cohorts, kernel=kernel)
    # interior point near the truth (far points floor some cells, where FD is meaningless)
    mixed = 0.8 * truth + 0.2 * sim.decode(np.full(sim.theta_dim, 0.5))
    theta = sim.encode(mixed).astype(float)
    ll, g = lik.log_likelihood(theta)
    assert np.isfinite(ll) and g.shape == (sim.theta_dim,)
    eps, tol = 1e-5, 1e-4 * np.abs(g).max()
    for i in range(sim.theta_dim):
        e = np.zeros_like(theta)
        e[i] = eps
        fd = lik.log_likelihood(theta + e, False)[0] - lik.log_likelihood(theta - e, False)[0]
        assert g[i] == pytest.approx(fd / (2 * eps), rel=1e-4, abs=tol)
    # posterior in u = logit(theta): gradient consistent too
    u = np.log(theta) - np.log1p(-theta)
    lp, gu = lik.log_posterior_u(u)
    e = np.zeros_like(u)
    e[0] = eps
    fd = (lik.log_posterior_u(u + e)[0] - lik.log_posterior_u(u - e)[0]) / (2 * eps)
    assert gu[0] == pytest.approx(fd, rel=1e-4, abs=1e-4 * np.abs(gu).max())


def test_unsupported_configurations_raise(law):
    obs2 = ObservationModel(("A", "B"))
    with pytest.raises(ValueError, match="galton_watson"):
        check_supported(BellmanHarris(law, lifetime_shape=8.0, lifetime_mean_bounds=[10, 80]), obs2)
    hetero = GaltonWatson(load_law(clone_heterogeneity={"kappa_bounds": [1, 100]}))
    with pytest.raises(ValueError, match="heterogeneity"):
        check_supported(hetero, obs2)
    three = GaltonWatson(load_law("three_type"))
    with pytest.raises(ValueError, match="two cell types"):
        check_supported(three, ObservationModel(three.types))
    # free founder rows are supported (tests/test_free_founder.py)
    check_supported(GaltonWatson(load_law(fixed={})), obs2)


def test_nuts_samples_a_correlated_gaussian():
    cov = np.array([[1.0, 0.8], [0.8, 2.0]])
    prec = np.linalg.inv(cov)

    def f(q):
        return -0.5 * q @ prec @ q, -prec @ q

    rng = np.random.default_rng(0)
    chains = [run_chain(f, rng.normal(size=2), 300, 800, np.random.default_rng(c))
              for c in range(3)]  # fmt: skip
    x = np.concatenate([c.draws for c in chains])
    np.testing.assert_allclose(x.mean(0), 0, atol=0.15)
    np.testing.assert_allclose(np.cov(x.T), cov, atol=0.3)
    stacked = np.stack([c.draws[:, 0] for c in chains])
    assert split_rhat(stacked) < 1.05 and ess(stacked) > 300
    assert sum(c.divergences for c in chains) == 0


def test_pipeline_run_writes_standard_artifacts(tmp_path):
    from bio_sbi.collect import collect
    from bio_sbi.pipeline import build_components, run_experiment

    from .test_pipeline import config, save_hydra_config

    cfg = config(
        "inference=exact_gw", "law=reversible_ab", "design.clones=8", "design.horizons=[2,4]",
        "inference.method.sbc.max_points=1e4", "inference.method.log_every=0",
        "inference.method.chains=2", "inference.method.warmup=20", "inference.method.draws=20",
        "inference.method.init_starts=2", "inference.method.init_maxiter=20",
        "inference.method.threads=2", "inference.sbc_cases=2", "inference.method.sbc.warmup=10",
        "inference.method.sbc.draws=10", "inference.method.sbc.init_starts=2",
        "budget.ppc_samples=5", "budget.sbc_samples=10",
    )  # fmt: skip
    out = tmp_path / "exact"
    save_hydra_config(cfg, out)
    metrics = run_experiment(cfg, out)
    for name in ("posterior.pt", "metrics.json", "recovery.csv", "posterior_samples.csv.gz",
                 "reference.npz", "ppc.npz", "sbc.npz", "figures/posterior.png"):  # fmt: skip
        assert (out / name).exists(), name
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["inference"] == "exact_gw" and saved["posterior_support_mass"] == 1.0
    assert saved["posterior_samples_drawn"] == 40
    assert saved["exact_gw"]["sampler"] == "nuts" and "B.to_A" in saved["exact_gw"]["rhat"]
    assert (
        saved["calibration"]["cases"]
        + saved["calibration"]["invalid_cases"]
        + saved["calibration"]["excluded_leakage"]
        == 2
    )
    assert metrics["ppc"]["features"] > 0
    runs, _ = collect(tmp_path)
    assert len(runs) == 1 and runs["inference"][0] == "exact_gw"
    # Unsupported inference model fails with a clear message.
    bad = config("inference=exact_gw", "simulator=bellman_harris", "budget.sbc_cases=0")
    with pytest.raises(ValueError, match="galton_watson"):
        run_experiment(bad, tmp_path / "bad")
    assert build_components(cfg).method.exact


def test_almost_undetectable_parameters_are_not_spuriously_likely(rng):
    """Where detection is ~impossible, floored cells / tiny P(detected) must not blow up."""
    law = load_law()
    sim = GaltonWatson(law)
    truth = sim.physical_from_mapping(law["reference"])
    obs = ObservationModel(sim.types)
    cohorts = [Cohort(h, sim.simulate_detected(truth, h, 256, np.ones(2), rng)) for h in (4, 10)]
    lik = ExactLikelihood(sim, obs, cohorts)
    # P(detected at gen 10) ~ 1e-25: without the detection floor this scored ~ +5000
    dying = {"A": {"death": 0.999, "div_AA": 0.0005, "div_AB": 0.0005},
             "B": {"death": 0.999, "to_A": 0.001}}  # fmt: skip
    far = sim.encode(sim.physical_from_mapping(dying)).astype(float)
    near = sim.encode(0.9 * truth + 0.1 * sim.decode(np.full(sim.theta_dim, 0.5)))
    ll_far = lik.log_likelihood(far, False)[0]
    assert ll_far < lik.log_likelihood(near.astype(float), False)[0] and ll_far < 0
    adj = lik.with_settings()
    adj.kernel = "adjoint"
    assert adj.log_likelihood(far, False)[0] == pytest.approx(ll_far, rel=1e-12)
    assert np.all(np.isfinite(adj.log_likelihood(far)[1]))


# ---------------------------------------------------------------- kernel=adjoint
OBSERVATIONS = {
    "perfect": {},
    "confusion": {"capture": [0.7, 0.9], "confusion": [[0.9, 0.1], [0.2, 0.8]]},
    "all": {"selection": "all", "capture": [0.8, 0.8]},
}


@pytest.mark.parametrize(
    "law_name,obs_name",
    [("two_type", "perfect"), ("reversible_ab", "perfect"), ("reversible_ab", "confusion"),
     ("two_type", "all"), ("minimal_ab", "perfect"), ("hierarchy_ab", "confusion")],
)  # fmt: skip
def test_adjoint_kernel_matches_loop(law_name, obs_name, rng):
    from scipy.special import expit, logit

    spec = load_law(law_name)
    sim = GaltonWatson(spec)
    truth = sim.physical_from_mapping(spec["reference"])
    obs = ObservationModel(sim.types, **OBSERVATIONS[obs_name])
    cohorts = []
    for h in (3, 5, 7):
        counts = obs.measure(sim.simulate(truth, h, 400, rng), rng)
        if obs.selection == "detected":
            counts = counts[counts.sum(1) > 0]
        cohorts.append(Cohort(h, counts[:100]))
    loop = ExactLikelihood(sim, obs, cohorts)
    adj = ExactLikelihood(sim, obs, cohorts, kernel="adjoint")
    # well-conditioned points (logit-normal around the truth): every observed cell well
    # above the FFT round-off, where both gradients are accurate
    th = sim.encode(truth).astype(float)
    for theta in expit(logit(th)[None] + rng.normal(0, 0.15, size=(8, len(th)))):
        v0, g0 = loop.log_likelihood(theta)
        v1, g1 = adj.log_likelihood(theta)
        assert v1 == pytest.approx(v0, rel=1e-12, abs=1e-12)
        assert adj.log_likelihood(theta, False)[0] == v1  # objective-only: same value
        np.testing.assert_allclose(g1, g0, rtol=1e-9, atol=1e-9 * np.abs(g0).max())
    u = logit(np.clip(theta, 1e-6, 1 - 1e-6))
    np.testing.assert_allclose(adj.log_posterior_u(u)[1], loop.log_posterior_u(u)[1],
                               rtol=1e-9, atol=1e-9 * np.abs(g0).max())  # fmt: skip


def test_adjoint_kernel_finite_differences_and_shared_monomials(rng):
    """Synthetic law with two outcomes sharing daughters, founder sizes 1-3, damping."""
    from bio_sbi.inference.gw_exact import (
        HorizonData,
        _detection_nb,
        histogram,
        law_poly,
        monomials,
        poly_coefficients,
    )
    from bio_sbi.inference.gw_exact import log_likelihood as gw_loglik

    off = np.array([[[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [1, 0]],
                    [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [0, 0]]])  # fmt: skip
    model = GWModel(off, np.array([6, 5]), np.array([0.8, 0.9]),
                    np.array([[0.95, 0.05], [0.1, 0.9]]), np.array([0.7, 0.3]),
                    np.array([1, 2, 3]), np.array([0.5, 0.3, 0.2]))  # fmt: skip
    mono = monomials(model)
    assert len(mono[0]) == 6 and mono[2][0, 5] == mono[2][0, 1]

    def probs(x):
        P = np.zeros((2, 6))
        P[0] = np.exp(x[:6]) / np.exp(x[:6]).sum()
        P[1, :5] = np.exp(x[6:]) / np.exp(x[6:]).sum()
        return P

    # directions: d P / d x_i of the softmax rows (row 0: 6 coordinates, row 1: 5)
    x0 = rng.normal(0, 0.3, 11)
    P = probs(x0)
    dP = np.zeros((11, 6))
    rows = np.array([0] * 6 + [1] * 5)
    for i in range(6):
        dP[i] = P[0] * ((np.arange(6) == i) - P[0, i])
    for i in range(5):
        dP[6 + i, :5] = P[1, :5] * ((np.arange(5) == i) - P[1, i])
    gens, grid = 5, grid_spec(40, 40)
    C, _ = poly_coefficients(mono[2], len(mono[0]), P, dP, rows)
    p_loop, _ = law_and_jac(model, P, dP[:0], rows[:0], gens, grid)
    p_poly, traj = law_poly(model, C, mono, gens, grid)
    assert traj.size == 0
    # same law up to round-off amplified by the damping (<= 1e4)
    np.testing.assert_allclose(p_poly, p_loop, rtol=1e-10, atol=1e-11)
    # detection probability: Numba port of detection_and_grad
    q0, dq0 = detection_and_grad(model, P, dP, rows, gens)
    q1, dq1 = _detection_nb(model.offspring, model.n_out, model.capture, P, dP, rows, gens,
                            model.founder, model.sizes, model.size_probs)  # fmt: skip
    assert q1 == pytest.approx(q0, rel=1e-13)
    np.testing.assert_allclose(dq1, dq0, rtol=1e-12, atol=1e-15)
    # data drawn from the law itself (cells comfortably above round-off)
    cond = np.clip(p_loop[:30, :30], 0, None)
    cond[0, 0] = 0
    cells = rng.choice(cond.size, size=300, p=(cond / cond.sum()).ravel())
    data = [HorizonData(gens, *histogram(np.c_[np.unravel_index(cells, cond.shape)]), grid)]
    v0, g0 = gw_loglik(model, P, dP, rows, data)
    v1, g1 = gw_loglik(model, P, dP, rows, data, kernel="adjoint")
    assert v1 == pytest.approx(v0, rel=1e-12)
    np.testing.assert_allclose(g1, g0, rtol=1e-9, atol=1e-9 * np.abs(g0).max())
    eps = 1e-6
    for i in range(11):
        e = np.zeros(11)
        e[i] = eps
        hi = gw_loglik(model, probs(x0 + e), dP[:0], rows[:0], data, kernel="adjoint")[0]
        lo = gw_loglik(model, probs(x0 - e), dP[:0], rows[:0], data, kernel="adjoint")[0]
        assert g1[i] == pytest.approx((hi - lo) / (2 * eps), rel=1e-5,
                                      abs=1e-6 * np.abs(g1).max())  # fmt: skip
    with pytest.raises(ValueError, match="kernel"):
        gw_loglik(model, P, dP, rows, data, kernel="batched")


def test_kernel_option_reaches_the_likelihood():
    from bio_sbi.inference import ExactGW

    assert ExactGW().kernel == "loop"
    sim = GaltonWatson(load_law("reversible_ab"))
    truth = sim.physical_from_mapping(load_law("reversible_ab")["reference"])
    rng = np.random.default_rng(0)
    cohorts = [Cohort(4, sim.simulate_detected(truth, 4, 32, np.ones(2), rng))]
    lik = ExactGW(kernel="adjoint").likelihood(sim, ObservationModel(sim.types), cohorts)
    assert lik.kernel == "adjoint" and lik.with_settings(floor=1e-14).kernel == "adjoint"
    with pytest.raises(ValueError, match="kernel"):
        ExactLikelihood(sim, ObservationModel(sim.types), cohorts, kernel="gpu")
