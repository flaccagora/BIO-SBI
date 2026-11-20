"""Expected Fisher information of the exact GW clone law and the Jeffreys prior (gw_fisher)."""

import numpy as np
import pytest
from scipy.special import expit, logit

from bio_sbi.inference.gw_fisher import (
    JeffreysPrior,
    batch_information,
    grid_size,
    horizon_information,
    support_bound,
    total_count_law,
)
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import GaltonWatson

from .conftest import load_law

#: Every outcome probability distinct and interior.
DISTINCT = {
    "A": {"death": 0.04, "persist": 0.07, "to_B": 0.11, "div_AA": 0.42, "div_AB": 0.15,
          "div_BB": 0.21},
    "B": {"death": 0.13, "persist": 0.05, "to_A": 0.36, "div_AA": 0.19, "div_AB": 0.08,
          "div_BB": 0.19},
}  # fmt: skip


def _setup(horizons, selection="detected"):
    sim = GaltonWatson(load_law("two_type"))
    obs = ObservationModel(sim.types, selection=selection)
    prior = JeffreysPrior(sim, obs, horizons)
    theta = sim.encode(sim.physical_from_mapping(DISTINCT)).astype(float)
    return sim, prior, theta


def _stick_jacobian(prior, theta):
    """``dP/dtheta`` of the A row, ``(6, 5)`` (outcome x A stick)."""
    _, dP, rows = prior.lik.jacobian(theta)
    return dP[rows == 0][:, :6].T


@pytest.mark.parametrize("selection", ["all", "detected"])
def test_one_generation_is_a_multinomial(selection):
    """Generation 1 from one A cell: the counts reveal A's outcome (death -> (0, 0)).

    All clones: I = J^T diag(1/p) J (multinomial over the 6 outcomes of A). Detected
    clones: the multinomial of the 5 visible outcomes renormalised by 1 - death. The B
    row is not identified (zero block).
    """
    sim, prior, theta = _setup([1], selection)
    info = prior.information(theta).info
    P, _, _ = prior.lik.jacobian(theta)
    p, J = P[0, :6], _stick_jacobian(prior, theta)
    if selection == "all":
        expected = J.T @ np.diag(1 / p) @ J
    else:
        death = int(np.flatnonzero(sim.law.offspring[0].sum(1) == 0)[0])  # no daughters
        vis = np.arange(6) != death
        q, dq = 1 - p[death], -J[death]
        pc = p[vis] / q
        dpc = J[vis] / q - np.outer(p[vis], dq) / q**2
        expected = dpc.T @ np.diag(1 / pc) @ dpc
    np.testing.assert_allclose(info[:5, :5], expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(info[5:, :], 0.0, atol=1e-12)


def test_one_generation_all_clones_jeffreys_is_dirichlet_half():
    """Fully observed multinomial row: Jeffreys = Dirichlet(1/2) (A block, up to a constant)."""
    _, prior, theta = _setup([1], "all")
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(4):
        t = theta.copy()
        t[:5] = expit(logit(theta[:5]) + rng.normal(0, 0.5, 5))
        info = prior.information(t).info[:5, :5]
        a, b = prior.stick_beta(0.5)
        log_dir = np.sum((a[:5] - 1) * np.log(t[:5]) + (b[:5] - 1) * np.log1p(-t[:5]))
        diffs.append(0.5 * np.linalg.slogdet(info)[1] - log_dir)
    assert np.ptp(diffs) < 1e-8


def test_score_matches_finite_differences():
    """d log p~_c / d theta of the conditional law vs central differences, high-mass cells."""
    sim, prior, theta = _setup([3])
    model, g = prior.model, 3
    P, dP, rows = prior.lik.jacobian(theta)
    L, _ = grid_size(model, P, g)
    hi = horizon_information(model, P, dP, rows, g)

    def cond_law(t):
        from bio_sbi.inference.gw_exact import GridSpec, detection_and_grad, law_and_jac

        P_, dP_, r_ = prior.lik.jacobian(t)
        p, dp = law_and_jac(model, P_, dP_, r_, g, GridSpec(L, L, 1.0, 1.0))
        q, dq = detection_and_grad(model, P_, dP_, r_, g)
        p[0, 0] = 0.0
        dp[:, 0, 0] = 0.0
        return p / q, dp / p[None].clip(1e-300) - (dq / q)[:, None, None]

    pc, score = cond_law(theta)
    cells = np.argwhere(pc > 1e-3)
    h = 1e-6
    for d in range(prior.dim):
        e = np.zeros_like(theta)
        e[d] = h
        hi_, lo_ = cond_law(theta + e)[0], cond_law(theta - e)[0]
        for a, b in cells:
            fd = (np.log(hi_[a, b]) - np.log(lo_[a, b])) / (2 * h)
            assert score[d, a, b] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    # the information is the p~-weighted outer product of these scores
    keep = pc > 1e-14
    s = score[:, keep]
    np.testing.assert_allclose(hi.info, (s * pc[keep]) @ s.T, rtol=1e-10)


def test_information_matches_monte_carlo():
    """E[s s^T] over simulated (detected) clones agrees with the exact information."""
    sim, prior, theta = _setup([3])
    model, g = prior.model, 3
    P, dP, rows = prior.lik.jacobian(theta)
    hi = horizon_information(model, P, dP, rows, g)
    from bio_sbi.inference.gw_exact import GridSpec, detection_and_grad, law_and_jac

    p, dp = law_and_jac(model, P, dP, rows, g, GridSpec(hi.grid, hi.grid, 1.0, 1.0))
    q, dq = detection_and_grad(model, P, dP, rows, g)
    rng = np.random.default_rng(1)
    counts = sim.simulate(sim.decode(theta), 3, 40_000, rng)
    counts = counts[counts.sum(1) > 0]
    s = dp[:, counts[:, 0], counts[:, 1]] / p[counts[:, 0], counts[:, 1]] - (dq / q)[:, None]
    mc = s @ s.T / s.shape[1]
    # Monte Carlo standard error of each entry
    se = np.sqrt(((s[:, None] * s[None]) ** 2).mean(-1) / s.shape[1])
    assert np.all(np.abs(mc - hi.info) < 5 * se + 1e-9)
    assert np.all(np.abs(s.mean(1)) < 5 * s.std(1) / np.sqrt(s.shape[1]))  # E[score] = 0


def test_grid_holds_the_law_and_matches_full_support():
    """Adaptive grid: tail of N below tolerance; same information as the full support."""
    sim, prior, theta = _setup([6])
    model, g = prior.model, 6
    P, dP, rows = prior.lik.jacobian(theta)
    pN = total_count_law(model, P, g)
    assert len(pN) >= support_bound(model, g) and pN.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(pN > -1e-14)
    L, tail = grid_size(model, P, g, tail_tol=1e-10)
    assert tail <= 1e-10
    small = horizon_information(model, P, dP, rows, g, tail_tol=1e-10)
    full = horizon_information(model, P, dP, rows, g, grid=support_bound(model, g))
    np.testing.assert_allclose(small.info, full.info, rtol=1e-7, atol=1e-10)
    assert small.omitted_mass < 1e-9


def test_jeffreys_density_is_finite_and_positive_definite():
    sim, prior, theta = _setup([2, 3])
    rng = np.random.default_rng(2)
    for _ in range(3):
        t = expit(logit(theta) + rng.normal(0, 0.3, theta.shape))
        res = prior.information(t)
        assert res.min_eigenvalue > 0
        assert np.isfinite(prior.log_density(t))
    a, b = prior.stick_beta(1.0)  # flat Dirichlet = the likelihood object's default prior
    np.testing.assert_allclose(a, prior.lik.prior_a)
    np.testing.assert_allclose(b, prior.lik.prior_b)
    # Dirichlet density in theta = Dirichlet density of the free probabilities x |dP/dtheta|
    from scipy.stats import dirichlet

    P, dP, rows = prior.lik.jacobian(theta)
    expected = 0.0
    for k in range(2):
        jac = dP[rows == k][:, :5]  # (stick, first 5 outcomes of row k)
        expected += dirichlet.logpdf(P[k, :6], np.full(6, 0.5))
        expected += np.log(abs(np.linalg.det(jac)))
    assert prior.log_dirichlet(theta, 0.5) == pytest.approx(expected, rel=1e-10)


def test_batched_torch_matches_numba():
    pytest.importorskip("torch")
    sim, prior, theta = _setup([2, 4])
    rng = np.random.default_rng(3)
    ts = expit(logit(theta) + rng.normal(0, 0.3, (3, theta.size)))
    res = batch_information(prior, ts, device="cpu", backend="torch")
    for t, r in zip(ts, res, strict=True):
        np.testing.assert_allclose(r.info, prior.information(t).info, rtol=1e-9, atol=1e-12)
