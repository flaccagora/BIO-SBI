"""Model evidence (bridge sampling, tempered-SMC path) and held-out predictive density.

Evidences are checked against closed forms (conjugate Gaussian) and brute-force
quadrature of the exact GW / BH posteriors with two free parameters (non-uniform Dirichlet
concentration, so the prior's normalising constant matters).
"""

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist
from scipy.stats import multivariate_normal as mvn

from bio_sbi.inference.evidence import (
    bridge_sampling,
    log_jac_probit,
    logit_from_probit,
    prior_log_norm,
    probit_from_logit,
)
from bio_sbi.inference.exact_gw import ExactGW
from bio_sbi.inference.heldout import lpd_summary, split_cohorts
from bio_sbi.inference.smc import tempered_smc
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import BellmanHarris, GaltonWatson
from bio_sbi.summaries import Cohort

from .conftest import load_law


def gaussian_toy(d=3):
    s0 = 2.0
    S = np.array([[1, 0.5, 0], [0.5, 1, 0.2], [0, 0.2, 0.5]]) * 0.05
    y = np.array([1.0, -0.5, 0.3])
    log_z = mvn(np.zeros(d), s0**2 * np.eye(d) + S).logpdf(y)
    P = np.linalg.inv(np.eye(d) / s0**2 + np.linalg.inv(S))
    m = P @ np.linalg.solve(S, y)

    def target(X):
        X = np.atleast_2d(X)
        return (mvn(np.zeros(d), s0**2 * np.eye(d)).logpdf(X).reshape(-1)
                + mvn(y, S).logpdf(X).reshape(-1))  # fmt: skip

    return target, log_z, m, P


def test_bridge_sampling_known_evidence():
    target, log_z, m, P = gaussian_toy()
    rng = np.random.default_rng(0)
    # iid posterior draws arranged as 4 chains
    X = mvn(m, P).rvs(4000, random_state=1).reshape(4, 1000, 3)
    r = bridge_sampling(X, target(X.reshape(-1, 3)).reshape(4, 1000), target, rng, 1000)
    assert r["mc_se"] < 0.02
    assert abs(r["log_evidence"] - log_z) < 4 * r["mc_se"] + 1e-3
    # autocorrelated chains (AR(1) moves) report a larger error than iid draws
    L = np.linalg.cholesky(P)
    z = rng.standard_normal((4, 1000, 3))
    for t in range(1, 1000):
        z[:, t] = 0.95 * z[:, t - 1] + np.sqrt(1 - 0.95**2) * z[:, t]
    Y = m + z @ L.T
    r2 = bridge_sampling(Y, target(Y.reshape(-1, 3)).reshape(4, 1000), target, rng, 1000)
    assert r2["ess_post"] < 0.2 * 4000
    assert abs(r2["log_evidence"] - log_z) < 4 * r2["mc_se"] + 1e-3


def test_tempered_smc_evidence_from_a_laplace_start():
    """Geometric path from a normalised (shifted, widened) start q: no correction needed."""
    target, log_z, m, P = gaussian_toy()
    ref = mvn(m + 0.05, 4 * P)
    res = tempered_smc(target, lambda r, n: ref.rvs(n, random_state=r).reshape(n, 3),
                       lambda U: ref.logpdf(np.atleast_2d(U)).reshape(-1),
                       np.random.default_rng(3), chains=32, steps=8, target_ess=200,
                       max_evals=20000)  # fmt: skip
    assert abs(res.info["log_evidence"] - log_z) < 0.25


def test_probit_coordinates_and_prior_constant():
    rng = np.random.default_rng(0)
    u = rng.normal(scale=6, size=(50, 3))
    np.testing.assert_allclose(logit_from_probit(probit_from_logit(u)), u, atol=1e-8)
    v = probit_from_logit(u)
    h = 1e-6
    num = (logit_from_probit(v + h) - logit_from_probit(v - h)) / (2 * h)
    np.testing.assert_allclose(log_jac_probit(v), np.log(num).sum(1), rtol=1e-5)
    # Beta(a, b) in logit coordinates: exp(a log s + b log(1 - s) + const) integrates to 1
    a, b = np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 0.5])
    grid = np.linspace(-60, 60, 200001)
    from scipy.special import log_expit

    for j in range(3):
        f = a[j] * log_expit(grid) + b[j] * log_expit(-grid) + prior_log_norm(a[j], b[j])
        assert np.trapezoid(np.exp(f), grid) == pytest.approx(1.0, rel=1e-6)


def _gw_toy(rng):
    law = load_law("two_type")
    law["fixed"] = {
        "founder": {"A": 1.0},
        "A": {"persist": 0.05, "to_B": 0.05, "div_AB": 0.05},
        "B": {"death": 0.2, "persist": 0.1, "to_A": 0.5, "div_AA": 0.1, "div_AB": 0.05,
              "div_BB": 0.05},
    }  # fmt: skip
    law["concentration"] = {"A": {"div_AA": 3.0, "death": 0.7}}  # non-uniform Dirichlet
    sim = GaltonWatson(law)
    truth = sim.physical_from_mapping({"A": {"death": 0.15, "div_AA": 0.5, "div_BB": 0.2}})
    cohorts = [Cohort(n, sim.simulate_detected(truth, n, 40, np.ones(2), rng)) for n in (3, 5)]
    return sim, cohorts


def test_exact_gw_evidence_matches_quadrature(rng):
    sim, cohorts = _gw_toy(rng)
    obs = ObservationModel(sim.types)
    method = ExactGW(chains=4, warmup=200, draws=500, threads=4, evidence_draws=1500, seed=2)
    lik = method.likelihood(sim, obs, cohorts)
    assert lik.dim == 2
    theta, diag = method.run(lik, evidence=True)
    ev = diag["evidence"]
    # brute force over the sticks s in (0, 1)^2 with an independent Beta prior density
    nodes, weights = np.polynomial.legendre.leggauss(160)
    s = 0.5 * (nodes + 1)
    wts = 0.5 * weights
    S = np.stack(np.meshgrid(s, s, indexing="ij"), -1).reshape(-1, 2)
    W = np.outer(wts, wts).reshape(-1)
    ll = np.array([lik.log_likelihood(t, grad=False)[0] for t in S])
    lprior = sum(beta_dist(lik.prior_a[j], lik.prior_b[j]).logpdf(S[:, j]) for j in range(2))
    log_z = logsumexp(ll + lprior, b=W)
    assert ev["mc_se"] < 0.05
    assert abs(ev["log_evidence"] - log_z) < 4 * ev["mc_se"] + 0.01


def test_exact_bh_evidence_matches_quadrature(rng):
    from bio_sbi.inference.exact_bh import ExactBH

    from .test_exact_bh import BOUNDS, detected

    law = load_law("reversible_ab")
    law["fixed"] = {
        "founder": {"A": 1.0},
        "A": {"persist": 0.05, "to_B": 0.05, "div_AB": 0.05, "div_BB": 0.25},
        "B": {"death": 0.1, "persist": 0.1, "to_A": 0.8, "div_AA": 0, "div_AB": 0, "div_BB": 0},
    }
    law["concentration"] = {"A": {"div_AA": 3.0}}
    sim = BellmanHarris(law, lifetime_shape=3.0, lifetime_mean_bounds=BOUNDS,
                        fixed={"B.mean_lifetime": 15.0})  # fmt: skip
    truth = sim.physical_from_mapping({"A": {"death": 0.1, "div_AA": 0.5},
                                       "A.mean_lifetime": 12.0})  # fmt: skip
    obs = ObservationModel(sim.types)
    cohorts = [Cohort(h, detected(sim, truth, h, 60, rng)) for h in (24.0, 48.0)]
    method = ExactBH(threads=4, init_candidates=4, chains=32, steps=6, seed=1, target_ess=300,
                     evidence_draws=1000)  # fmt: skip
    lik = method.likelihood(sim, obs, cohorts)
    theta, w, diag = method.run(lik)
    ev = diag["evidence"]
    assert ev["prior_log_norm"] == pytest.approx(np.log(3.0))  # Beta(1, 3) stick
    u = lik.u_from_theta(theta)
    mu, sd = u.mean(0), u.std(0)
    axes = [np.linspace(mu[j] - 9 * sd[j], mu[j] + 9 * sd[j], 81) for j in range(2)]
    U = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    lp = lik.log_posterior_batch(U)
    lik.close()
    log_z = logsumexp(lp) + np.log((axes[0][1] - axes[0][0]) * (axes[1][1] - axes[1][0]))
    log_z += ev["prior_log_norm"]
    assert abs(ev["log_evidence"] - log_z) < 4 * ev["mc_se"] + 0.02
    assert abs(ev["log_evidence_smc"] - log_z) < 0.3  # SMC path (Laplace start), no MC error


def test_holdout_split_and_lpd(rng):
    cohorts = [Cohort(144.0, rng.integers(1, 50, size=(165, 2))),
               Cohort(240.0, rng.integers(1, 50, size=(101, 2)))]  # fmt: skip
    train, test = split_cohorts(cohorts, 0.2, seed=3)
    assert [len(c.counts) for c in test] == [33, 20]
    assert [len(c.counts) for c in train] == [132, 81]
    for c, tr, te in zip(cohorts, train, test, strict=True):
        both = np.concatenate([tr.counts, te.counts])
        assert sorted(map(tuple, both)) == sorted(map(tuple, c.counts))
    # same seed -> same split; another seed -> another split
    again = split_cohorts(cohorts, 0.2, seed=3)[1]
    other = split_cohorts(cohorts, 0.2, seed=4)[1]
    assert np.array_equal(again[0].counts, test[0].counts)
    assert not np.array_equal(other[0].counts, test[0].counts)
    # lpd = log-mean-exp of the per-draw test log-likelihoods
    ll = np.array([-10.0, -11.0, -12.0])
    s = lpd_summary(ll, 5)
    assert s["lpd"] == pytest.approx(np.log(np.mean(np.exp(ll))))
    assert s["lpd_per_clone"] == pytest.approx(s["lpd"] / 5)
    assert lpd_summary(np.full(10, -3.0), 2)["lpd_se"] == pytest.approx(0.0)


def test_holdout_lpd_of_exact_gw_matches_direct_computation(rng):
    sim, cohorts = _gw_toy(rng)
    obs = ObservationModel(sim.types)
    method = ExactGW(chains=2, warmup=100, draws=150, threads=2, evidence_draws=200,
                     holdout={"fraction": 0.25, "seed": 1, "draws": 100})  # fmt: skip
    post = method.fit_data(sim, obs, cohorts, x_dim=4)
    ho = post.diagnostics["holdout"]
    assert ho["n_test"] == 20 and ho["n_train"] == [30, 30] and ho["draws"] == 100
    _, test = split_cohorts(cohorts, 0.25, 1)
    tlik = method.likelihood(sim, obs, test)
    ll = np.array([tlik.log_likelihood(t, grad=False)[0] for t in post.draws])
    # the lpd over 100 of the draws is close to the lpd over all draws
    assert ho["lpd"] == pytest.approx(logsumexp(ll) - np.log(len(ll)), abs=0.5)
    assert np.isfinite(post.diagnostics["evidence"]["log_evidence"])


def test_holdout_of_exact_bh_runs_on_the_split(rng):
    from bio_sbi.inference.exact_bh import ExactBH

    from .test_exact_bh import BOUNDS, detected

    sim = BellmanHarris(load_law("reversible_ab"), lifetime_shape=2.0, lifetime_mean_bounds=BOUNDS)
    truth = sim.physical_from_mapping({**load_law("reversible_ab")["reference"],
                                       "A.mean_lifetime": 12.0, "B.mean_lifetime": 18.0})  # fmt: skip
    cohorts = [Cohort(h, detected(sim, truth, h, 16, rng)) for h in (20.0, 36.0)]
    method = ExactBH(threads=2, chains=8, steps=3, min_final_steps=4, check_every=2,
                     max_evals=300, target_ess=50, init_candidates=2, opt_maxiter=4,
                     evidence_draws=64, holdout={"fraction": 0.25, "seed": 0, "draws": 20})  # fmt: skip
    post = method.fit_data(sim, ObservationModel(sim.types), cohorts, x_dim=4)
    ho = post.diagnostics["holdout"]
    assert ho["n_train"] == [12, 12] and ho["n_test"] == 8 and ho["draws"] == 20
    assert np.isfinite(ho["lpd"]) and ho["lpd"] < 0
    ev = post.diagnostics["evidence"]
    assert set(ev) >= {"log_evidence", "mc_se", "log_evidence_smc", "prior_log_norm"}
