"""Ridge-straightening sampler coordinates of exact_bh (``inference.method.reparam``)."""

import numpy as np
import pytest
from scipy.special import expit, log_expit
from scipy.stats import beta as beta_dist
from scipy.stats import kstest

from bio_sbi.inference.bh_reparam import (
    Reparam,
    malthusian,
    ridge_lifetime,
    stable_lifetimes,
    stable_ratio,
)
from bio_sbi.inference.exact_bh import BHLikelihood, ExactBH
from bio_sbi.inference.smc import ess_bulk, tempered_smc
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import BellmanHarris
from bio_sbi.summaries import Cohort

from .conftest import load_law

SPECS = ["growth", "stable", "probit", "stable+probit"]


@pytest.fixture(scope="module")
def lik():
    """reversible_ab (7 sticks + 2 log lifetimes), a small dataset."""
    rng = np.random.default_rng(3)
    law = load_law("reversible_ab")
    sim = BellmanHarris(law, lifetime_shape=4.0, lifetime_mean_bounds=[8.0, 40.0])
    ref = {**law["reference"], "A.mean_lifetime": 12.0, "B.mean_lifetime": 18.0}
    truth = sim.physical_from_mapping(ref)
    cohorts = []
    for h in (20.0, 40.0):
        c = sim.simulate(truth, h, 160, rng)
        cohorts.append(Cohort(h, c[c.sum(1) > 0][:40]))
    lk = BHLikelihood(sim, ObservationModel(sim.types), cohorts)
    lk.anchor = lk.u_from_theta(sim.encode(truth).astype(float))
    yield lk
    lk.close()


def test_malthusian_parameter():
    rng = np.random.default_rng(0)
    M = rng.uniform(0, 1.5, size=(50, 2, 2))
    tau = rng.uniform(5, 40, size=(50, 2))
    # exponential lifetimes (shape 1): Markov branching, alpha = top eigenvalue of
    # diag(1 / tau) (M - I)
    a = malthusian(M, tau, np.ones(2))
    G = (M - np.eye(2)) / tau[:, :, None]
    np.testing.assert_allclose(a, np.linalg.eigvals(G).real.max(1), rtol=1e-8, atol=1e-10)
    # no transitions between types, Gamma(m) lifetimes: alpha tau / m = a^(1/m) - 1
    D = M * np.eye(2)
    m = np.array([8.0, 3.0])
    expect = np.max(m * (np.diagonal(D, axis1=1, axis2=2) ** (1 / m) - 1) / tau, axis=1)
    np.testing.assert_allclose(malthusian(D, tau, m), expect, rtol=1e-8, atol=1e-10)


def test_ridge_lifetime_solves_the_growth_equation():
    rng = np.random.default_rng(1)
    n, m = 200, np.array([8.0, 4.0])
    M0 = rng.uniform(0, 1.2, size=(n, 2, 2))
    M0[:, 0, 0] += 0.6  # A self-renews
    tau_b = rng.uniform(10, 60, n)
    for alpha0, scale in ((0.02, 1.0), (-0.01, 0.5)):  # growing / shrinking clones
        M = M0 * scale
        lt, side = ridge_lifetime(M, tau_b, m[0], m[1], alpha0)
        ok = side == 0
        assert ok.sum() > 20 and np.all(np.isfinite(lt[ok]))
        tau = np.stack([np.exp(lt[ok]), tau_b[ok]], axis=1)
        np.testing.assert_allclose(malthusian(M[ok], tau, m), alpha0, rtol=1e-6, atol=1e-9)
        # no root: alpha - alpha0 keeps its sign over all lifetimes
        grid = np.geomspace(1e-3, 1e4, 200)
        for i in np.flatnonzero(~ok)[:10]:
            tau = np.stack([grid, np.full_like(grid, tau_b[i])], axis=1)
            diff = malthusian(np.repeat(M[i][None], len(grid), 0), tau, m) - alpha0
            assert np.all(diff > 0) or np.all(diff < 0)


@pytest.mark.parametrize("spec", SPECS)
def test_round_trip_and_jacobian(lik, spec):
    rp = Reparam(lik, spec, anchor_u=lik.anchor)
    rng = np.random.default_rng(2)
    U = lik.anchor + 2.0 * rng.standard_normal((40, lik.dim))
    V = rp.to_v(U)
    np.testing.assert_allclose(rp.to_u(V), U, atol=1e-8)
    np.testing.assert_allclose(rp.to_v(rp.to_u(V)), V, atol=1e-8)
    eps = 1e-6
    for v in V[:8]:
        J = np.empty((lik.dim, lik.dim))
        for j in range(lik.dim):
            e = np.zeros(lik.dim)
            e[j] = eps
            J[:, j] = (rp.to_u(v + e)[0] - rp.to_u(v - e)[0]) / (2 * eps)
        assert rp.log_jac(v)[0] == pytest.approx(np.linalg.slogdet(J)[1], abs=1e-5)
    sheared = {"growth": ["A.log_mean_lifetime"], "probit": [],
               "stable": ["A.log_mean_lifetime", "B.log_mean_lifetime"]}  # fmt: skip
    assert rp.info()["sheared"] == sheared[spec.split("+")[0]]
    # the shears only move the lifetimes (unit-triangular Jacobian)
    if "probit" not in spec:
        moved = np.any(np.abs(V - U) > 0, axis=0)
        assert [lik.space.theta_names[j] for j in np.flatnonzero(moved)] == rp.info()["sheared"]


def test_stable_lifetimes_solve_both_equations():
    rng = np.random.default_rng(7)
    n, m = 300, np.array([8.0, 4.0])
    M = rng.uniform(0, 1.0, size=(n, 2, 2))
    M[:, 0, 0] += 0.8
    M[:, 1, 1] *= 0.8  # B alone subcritical
    lt, ok = stable_lifetimes(M, m, 0.02, 0.6)
    assert ok.mean() > 0.3
    tau = np.exp(lt[ok])
    a = malthusian(M[ok], tau, m)
    np.testing.assert_allclose(a, 0.02, rtol=1e-6)
    np.testing.assert_allclose(stable_ratio(M[ok], tau, m, a), 0.6, rtol=1e-5)
    assert np.all(np.isnan(lt[~ok]))


@pytest.mark.parametrize("spec", ["growth", "stable"])
def test_shear_follows_the_ridge(lik, spec):
    """Fixing the sheared v keeps the Malthusian parameter (and the stable ratio) ~constant."""
    rp = Reparam(lik, spec, anchor_u=lik.anchor)
    assert rp.alpha(lik.anchor)[0] == pytest.approx(rp.alpha0)
    # the anchor lies on its own ridge
    np.testing.assert_allclose(rp.to_v(lik.anchor)[0, rp.cols], 0.0, atol=1e-6)
    rng = np.random.default_rng(4)
    U = lik.anchor + 0.3 * rng.standard_normal((200, lik.dim))
    U[:, rp.cols] = lik.anchor[rp.cols]  # u fixed
    V = rp.to_v(U)
    V[:, rp.cols] = 0.0  # v fixed (on the ridge)
    W = rp.to_u(V)
    a_u, a_v = rp.alpha(U), rp.alpha(W)
    assert np.std(a_v - rp.alpha0) < 0.05 * np.std(a_u - rp.alpha0)
    if spec == "stable":
        r_u, r_v = rp.ratio(U), rp.ratio(W)
        assert np.std(np.log(r_v / rp.r0)) < 0.05 * np.std(np.log(r_u / rp.r0))


@pytest.mark.parametrize("spec", SPECS)
def test_prior_is_preserved(lik, spec):
    """MCMC on the prior in v coordinates, mapped back: theta marginals = the Beta priors."""
    rp = Reparam(lik, spec, anchor_u=lik.anchor)

    def prior_u(U):
        return log_expit(U) @ lik.prior_a + log_expit(-U) @ lik.prior_b

    def logpost(V):
        V = np.atleast_2d(V)
        return prior_u(rp.to_u(V)) + rp.log_jac(V)

    c = rp.to_v(lik.anchor)[0]
    d = lik.dim

    def ref_sample(r, n):
        return c + 2.0 * r.standard_normal((n, d))

    def ref_logpdf(V):
        return -0.5 * (((np.atleast_2d(V) - c) / 2.0) ** 2).sum(1) - d * np.log(
            2.0 * np.sqrt(2 * np.pi)
        )

    res = tempered_smc(logpost, ref_sample, ref_logpdf, np.random.default_rng(5), chains=32,
                       steps=8, target_ess=1500, max_evals=400000, max_final_steps=20000,
                       monitor=rp.to_u)  # fmt: skip
    assert res.info["reached_posterior"] and res.info["min_ess_bulk"] >= 1500
    U = rp.to_u(res.draws)
    S = expit(U)  # unit-scaled theta ~ Beta(prior_a, prior_b) per coordinate
    thin = max(1, int(len(S) / res.info["min_ess_bulk"]))
    for j in range(d):
        x = S[::thin, j]
        p = kstest(x, beta_dist(lik.prior_a[j], lik.prior_b[j]).cdf).pvalue
        assert p > 1e-3, (spec, lik.space.theta_names[j], p)


def test_banana_shear_improves_mixing():
    """Toy ridge: y | x ~ N(x^2 - 1, 0.1^2). A shear v_y = y - (x^2 - 1) (unit Jacobian)
    makes the target Gaussian: correct moments and a much higher ESS per evaluation."""
    from scipy.stats import norm

    def logpost(U):
        U = np.atleast_2d(U)
        return norm.logpdf(U[:, 0], 0, 1) + norm.logpdf(U[:, 1], U[:, 0] ** 2 - 1, 0.1)

    def to_u(V):
        V = np.atleast_2d(V)
        return np.stack([V[:, 0], V[:, 1] + V[:, 0] ** 2 - 1], axis=1)

    def ref_sample(r, n):
        return 2.0 * r.standard_normal((n, 2))

    def ref_logpdf(U):
        return norm.logpdf(np.atleast_2d(U), 0, 2.0).sum(1)

    out = {}
    for name, lp, mon in (("plain", logpost, None), ("shear", lambda V: logpost(to_u(V)), to_u)):
        res = tempered_smc(lp, ref_sample, ref_logpdf, np.random.default_rng(6), chains=32,
                           steps=6, imh_prob=0.0, target_ess=1e9, max_evals=12000,
                           max_final_steps=10000, monitor=mon)  # fmt: skip
        X = res.chains if mon is None else to_u(res.draws).reshape(res.chains.shape)
        out[name] = min(ess_bulk(X[:, :, j]) for j in range(2)) / res.info["evals"]
        if mon is not None:  # the plain chains have not mixed at this budget
            D = X.reshape(-1, 2)
            assert abs(D[:, 0].mean()) < 0.1 and D[:, 0].std() == pytest.approx(1.0, rel=0.1)
            assert abs(D[:, 1].mean()) < 0.15  # E[x^2 - 1] = 0
        # the stopping rule's diagnostics are computed on the monitored coordinates
        assert res.info["min_ess_bulk"] == pytest.approx(min(ess_bulk(X[:, :, j])
                                                             for j in range(2)))  # fmt: skip
    assert out["shear"] > 3 * out["plain"], out


def test_exact_bh_reparam_options(lik):
    with pytest.raises(ValueError, match="unknown reparam"):
        ExactBH(reparam="whiten")
    with pytest.raises(ValueError, match="start=laplace"):
        ExactBH(reparam="growth", start="prior")
    with pytest.raises(ValueError, match="unknown reparam"):
        Reparam(lik, "whiten")
