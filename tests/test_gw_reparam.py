"""exact_gw sampler coordinates (``inference.method.reparam``, gw_reparam.py) and SMC start."""

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import beta as beta_dist
from scipy.stats import kstest

from bio_sbi.inference import ExactGW
from bio_sbi.inference.exact_gw import ExactLikelihood
from bio_sbi.inference.gw_reparam import StickCoordinates, brow_order, parse_spec
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import GaltonWatson
from bio_sbi.summaries import Cohort

from .conftest import load_law

SPECS = ["probit", "brow", "brow+probit"]
#: The flat_btoa prior of the XG2 preset: B row Dirichlet(to_A 1, others 0.2), A row flat.
FLAT_BTOA = {"B": {"death": 0.2, "persist": 0.2, "to_A": 1, "div_AA": 0.2, "div_AB": 0.2,
                   "div_BB": 0.2}}  # fmt: skip
SPARSE = {t: dict.fromkeys(["death", "persist", f"to_{o}", "div_AA", "div_AB", "div_BB"], 0.2)
          for t, o in (("A", "B"), ("B", "A"))}  # fmt: skip
TRUTH = {
    "A": {"death": 0.04, "persist": 0.07, "to_B": 0.11, "div_AA": 0.42, "div_AB": 0.15,
          "div_BB": 0.21},
    "B": {"death": 0.13, "persist": 0.05, "to_A": 0.36, "div_AA": 0.19, "div_AB": 0.08,
          "div_BB": 0.19},
}  # fmt: skip


def make_lik(concentration=None, clones=48, seed=0):
    sim = GaltonWatson(load_law(concentration=concentration or {}))
    truth = sim.physical_from_mapping(TRUTH)
    rng = np.random.default_rng(seed)
    obs = ObservationModel(sim.types)
    cohorts = [Cohort(h, sim.simulate_detected(truth, h, clones, np.ones(2), rng))
               for h in (3, 5)]  # fmt: skip
    return ExactLikelihood(sim, obs, cohorts, kernel="adjoint"), truth


@pytest.fixture(scope="module")
def lik():
    return make_lik()[0]


def jacobian(f, v, h=1e-6):
    """Central finite-difference Jacobian of ``f: R^d -> R^d`` (smooth maps only)."""
    cols = []
    for i in range(len(v)):
        e = np.zeros_like(v)
        e[i] = h
        cols.append((f(v + e) - f(v - e)) / (2 * h))
    return np.stack(cols, axis=1)


def interior_points(coords, rng, n, near_data=True):
    """``v`` of points kept away from the boundary (finite differences).

    ``near_data``: mixtures 0.8 truth + 0.2 random point (far from the data, observed cells
    have probabilities near the FFT accuracy, ~1e-15, and the log-likelihood is round-off
    noise of several nats, whatever the coordinates).
    """
    s = rng.uniform(0.05, 0.95, size=(n, coords.dim))
    if not near_data:
        return coords.from_sticks(s)
    space = coords.space
    truth = space.complete({k: v for r, vals in TRUTH.items()
                            for k, v in {f"{r}.{e}": p for e, p in vals.items()}.items()})  # fmt: skip
    phys = 0.8 * truth + 0.2 * space.decode(coords.theta(coords.from_sticks(s)))
    return coords.from_theta(np.stack([space.encode(p) for p in phys]).astype(float))


# ---------------------------------------------------------------- spec / bijection
def test_specs_and_orders(lik):
    assert parse_spec("none") == ("logit", False)
    assert parse_spec("brow+probit") == ("probit", True)
    with pytest.raises(ValueError, match="reparam"):
        parse_spec("cauchit")
    with pytest.raises(ValueError, match="reparam"):
        ExactGW(reparam="brow+cloglog")
    assert brow_order(["A", "B"]) == {"B": ["to_A", "persist", "death", "div_AA", "div_AB",
                                            "div_BB"]}  # fmt: skip
    assert StickCoordinates(lik, "none").identity
    c = StickCoordinates(lik, "brow")
    assert not c.identity and c.orders["B"][:3] == ["to_A", "persist", "death"]
    assert c.orders["A"] == ["death", "persist", "to_B", "div_AA", "div_AB", "div_BB"]
    # explicit stick order: listed entries first, the rest in table order
    c = StickCoordinates(lik, "none", {"A": ["div_AB"]})
    assert c.orders["A"] == ["div_AB", "death", "persist", "to_B", "div_AA", "div_BB"]
    with pytest.raises(ValueError, match="stick_order"):
        StickCoordinates(lik, "none", {"A": ["to_A"]})


@pytest.mark.parametrize("spec", SPECS)
def test_bijection_round_trip_and_physical(lik, spec, rng):
    c = StickCoordinates(lik, spec)
    theta = rng.uniform(0.02, 0.98, size=(50, lik.dim))
    v = c.from_theta(theta)
    np.testing.assert_allclose(c.theta(v), theta, atol=1e-12)
    np.testing.assert_allclose(c.from_theta(c.theta(v)), v, atol=1e-8)
    # physical parameters from v agree with the space's decode of theta(v)
    np.testing.assert_allclose(c.physical(v), lik.space.decode(c.theta(v)), atol=1e-14)
    # B.to_A is the first stick of B's row under brow
    if "brow" in spec:
        j = lik.space.names.index("B.to_A")
        k = c.blocks[1][4].start
        np.testing.assert_allclose(c.physical(v)[:, j], c.sticks(v[:, k])[0], atol=1e-14)


# ---------------------------------------------------------------- target unchanged
@pytest.mark.parametrize("spec", SPECS)
def test_log_density_is_original_times_jacobian(lik, spec, rng):
    """log p_v(v) = log p_u(u(v)) + log |det du/dv| (u = logit theta, the default target)."""
    c = StickCoordinates(lik, spec)
    for v in interior_points(c, rng, 5):

        def u_of(x):
            return logit(c.theta(x))

        u = u_of(v)
        logdet = np.linalg.slogdet(jacobian(u_of, v))[1]
        lp_v, _ = c.log_posterior(v)
        lp_u, _ = lik.log_posterior_u(u)
        assert lp_v == pytest.approx(lp_u + logdet, abs=1e-6)
        assert c.log_posterior_value(v) == pytest.approx(lp_v, abs=1e-9)


@pytest.mark.parametrize("concentration", [None, FLAT_BTOA, SPARSE])
@pytest.mark.parametrize("spec", SPECS)
def test_prior_density_is_original_times_jacobian(spec, concentration, rng):
    """Normalised prior of v = the space's stick Betas (as u) times the Jacobian."""
    lik, _ = make_lik(concentration, clones=4)
    c = StickCoordinates(lik, spec)
    a, b = lik.prior_a, lik.prior_b
    for v in interior_points(c, rng, 5, near_data=False):

        def u_of(x):
            return logit(c.theta(x))

        theta = c.theta(v)
        # density of u = logit(theta): Beta(theta) * theta (1 - theta)
        lp_u = (beta_dist.logpdf(theta, a, b) + np.log(theta) + np.log1p(-theta)).sum()
        logdet = np.linalg.slogdet(jacobian(u_of, v))[1]
        assert c.log_prior(v, normalised=True) == pytest.approx(lp_u + logdet, abs=1e-6)


@pytest.mark.parametrize("concentration", [None, FLAT_BTOA, SPARSE])
@pytest.mark.parametrize("spec", SPECS)
def test_prior_moments_are_the_dirichlet(spec, concentration):
    """Prior draws of v map to the same Dirichlet rows (means, covariances, a marginal)."""
    lik, _ = make_lik(concentration, clones=4)
    c = StickCoordinates(lik, spec)
    n = 40_000
    phys = c.physical(c.sample_prior(np.random.default_rng(1), n))
    space = lik.space
    for r in space.rows:
        if not r.n_sticks:
            continue
        cols = space.row_slice(r.prefix).start + np.asarray(r.free)
        alpha = np.asarray(r.alpha, float)
        a0 = alpha.sum()
        mean = alpha / a0
        cov = (np.diag(mean) - np.outer(mean, mean)) / (a0 + 1)
        x = phys[:, cols]
        se_mean = np.sqrt(np.diag(cov) / n)
        assert np.all(np.abs(x.mean(0) - mean) < 5 * se_mean + 1e-4)
        emp = np.cov(x, rowvar=False)
        # sd of a sample covariance ~ sqrt((E[x^2 y^2] + cov^2) / n): loose bound
        scale = np.sqrt(np.outer(np.diag(cov), np.diag(cov)) / n) * 3
        assert np.all(np.abs(emp - cov) < 5 * scale + 1e-4)
        # one marginal against its Beta law
        j = r.entries.index("to_A") if r.prefix == "B" else r.entries.index("div_AA")
        k = list(r.free).index(j)
        assert kstest(x[:, k], beta_dist(alpha[k], a0 - alpha[k]).cdf).pvalue > 1e-3


@pytest.mark.parametrize("spec", SPECS)
def test_gradient_is_the_chain_rule(lik, spec, rng):
    """grad log p_v = (d theta/dv)^T grad log L(theta) + grad (log prior + log Jacobian)."""
    c = StickCoordinates(lik, spec)
    for v in interior_points(c, rng, 3):
        _, g = c.log_posterior(v)
        _, g_theta = lik.log_likelihood(c.theta(v))
        J = jacobian(c.theta, v)
        prior_grad = jacobian(lambda x: np.atleast_1d(c.log_prior(x)), v, 1e-5)[0]
        expect = J.T @ g_theta + prior_grad
        np.testing.assert_allclose(g, expect, rtol=1e-6, atol=1e-6 * np.abs(expect).max())


def test_probit_prior_gradient_far_in_the_tails(lik):
    """Mills ratios stay finite where Phi underflows towards 0 / 1."""
    c = StickCoordinates(lik, "probit")
    v = np.linspace(-7.5, 7.5, lik.dim)
    lp, g = c.log_posterior(v)
    assert np.isfinite(lp) and np.all(np.isfinite(g))


# ---------------------------------------------------------------- sampler plumbing
def test_default_coordinates_are_the_original_code_path(lik):
    m = ExactGW()
    c = m.coordinates(lik)
    assert m.reparam == "none" and not m.smc["enabled"] and c.identity
    u = np.linspace(-2, 2, lik.dim)
    lp, g = c.log_posterior(u)
    lp0, g0 = lik.log_posterior_u(u)
    assert lp == lp0 and np.array_equal(g, g0)
    assert np.array_equal(c.theta(u), expit(u))


@pytest.mark.parametrize("spec", ["none", *SPECS])
def test_nuts_runs_in_every_coordinate_system(lik, spec):
    m = ExactGW(chains=2, warmup=30, draws=20, threads=2, init_starts=2, init_maxiter=30,
                log_every=0, seed=1, reparam=spec)  # fmt: skip
    theta, diag = m.run(lik)
    assert theta.shape == (40, lik.dim) and np.all((theta > 0) & (theta < 1))
    assert (diag.get("reparam", {"spec": "none"})["spec"]) == spec
    assert np.isfinite(diag["max_rank_rhat"]) and diag["gradient_evals"] > 0


def test_smc_start_and_smc_sampler(lik):
    smc = {"enabled": True, "chains": 16, "steps": 4, "max_evals": 6000, "warmup": 30,
           "final_steps": 4}  # fmt: skip
    m = ExactGW(chains=2, draws=20, threads=2, log_every=0, seed=2, reparam="brow+probit",
                smc=smc)  # fmt: skip
    theta, diag = m.run(lik, smc=m.smc)
    assert theta.shape == (40, lik.dim)
    assert diag["smc"]["reached_posterior"] and diag["smc"]["mode"] == "start"
    assert diag["smc_start"]["select"] == "resample" and diag["smc_start"]["warmup"] == 30
    assert m.last_smc.draws.shape == (16 * 5, lik.dim)  # population + final_steps states
    final = {**smc, "mode": "final", "target_ess": 50, "max_final_steps": 64}
    m = ExactGW(threads=2, log_every=0, seed=2, smc=final)
    theta, diag = m.run(lik, smc=m.smc)
    assert diag["sampler"] == "smc" and diag["gradient_evals"] == 0
    assert theta.shape[1] == lik.dim and np.all((theta > 0) & (theta < 1))
    with pytest.raises(ValueError, match="smc"):
        ExactGW(smc={"mode": "both"})
    with pytest.raises(ValueError, match="unknown options"):
        ExactGW(smc={"chain": 3})


@pytest.mark.slow
def test_posterior_moments_agree_across_coordinates():
    """Small easy design: brow+probit NUTS and the SMC sampler match the default sampler."""
    lik, _ = make_lik(clones=160, seed=4)
    runs = {}
    for name, kw in {"none": {}, "brow+probit": {"reparam": "brow+probit"},
                     "smc": {"smc": {"enabled": True, "mode": "final", "target_ess": 400}}
                     }.items():  # fmt: skip
        m = ExactGW(chains=4, warmup=200, draws=400, threads=4, log_every=0, seed=5, **kw)
        theta, diag = m.run(lik, smc=m.smc)
        phys = lik.space.decode(theta)
        free = [lik.space.names.index(n) for n in lik.space.free_names]
        ess_ = np.array([diag["ess_bulk"][lik.space.names[i]] for i in free])
        runs[name] = (phys[:, free].mean(0), phys[:, free].std(0), ess_)
    m0, s0, e0 = runs["none"]
    for name in ("brow+probit", "smc"):
        m1, s1, e1 = runs[name]
        z = (m1 - m0) / np.sqrt(s0**2 / e0 + s1**2 / e1)
        assert np.abs(z).max() < 4.5, (name, z)
        np.testing.assert_allclose(s1, s0, rtol=0.25)
