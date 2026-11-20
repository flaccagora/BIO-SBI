"""Likelihood-ratio test for dedifferentiation (bio_sbi.inference.dediff_lrt)."""

import numpy as np
import pytest
from scipy import stats

from bio_sbi.inference import dediff_lrt as D

A_ROW = {"death": 0.03, "persist": 0.03, "to_B": 0.05, "div_AA": 0.55, "div_AB": 0.05,
         "div_BB": 0.29}  # fmt: skip
NULL = D.P_from_rows(A_ROW, {"death": 0.1, "persist": 0.5, "div_BB": 0.4})
ALT = D.P_from_rows(A_ROW, {"death": 0.1, "persist": 0.2, "to_A": 0.3, "div_BB": 0.4})
HORIZONS = (4, 6)
FAST = D.FitSettings(h0_random=1, full_random=0, q_starts=(0.01,),
                     compositions=("to_A", "mixed"), pins=(0.3,))  # fmt: skip


@pytest.fixture(scope="module")
def sim():
    return D.make_simulator()


@pytest.fixture(scope="module")
def null_lik(sim):
    return D.likelihood(sim, D.simulate_counts(sim, NULL, HORIZONS, 96, 0, ("reference",)))


def test_parametrisations_fix_zeros_and_roundtrip():
    rng = np.random.default_rng(0)
    H0, FULL = D.h0_model(), D.full_model()
    for _ in range(20):
        P = H0.P(rng.uniform(0, 1, H0.dim))[0]
        assert np.all(P[1, D.B_DEDIFF] == 0.0)
        assert np.allclose(P.sum(1), 1.0)
        Q = D.random_P(rng, h0=False)
        assert np.allclose(FULL.P(FULL.encode(Q))[0], Q)
    pin = D.pinned_model(0.25)
    P = pin.P(rng.uniform(0, 1, pin.dim))[0]
    assert P[1, D.ENTRIES["B"].index("to_A")] == 0.25 and np.isclose(P[1].sum(), 1.0)


def test_simplex_derivative_exact_at_bounds():
    s = np.array([0.3, 1.0, 0.0, 0.6])
    _, d = D._simplex(s, 0.8)
    for j in range(len(s)):
        e = np.zeros(len(s))
        e[j] = 1e-7
        lo = s - e if s[j] > 0 else s
        hi = s + e if s[j] < 1 else s
        fd = (D._simplex(hi, 0.8)[0] - D._simplex(lo, 0.8)[0]) / (hi[j] - lo[j])
        assert np.allclose(fd, d[j], atol=1e-6)


def test_gradient_matches_finite_differences(null_lik):
    rng = np.random.default_rng(1)
    for par, P in ((D.h0_model(), NULL), (D.full_model(), ALT)):
        s = np.clip(par.encode(P) + rng.normal(0, 0.02, par.dim), 0.02, 0.98)
        _, g = D.loglik(null_lik, par, s)
        for j in range(par.dim):
            e = np.zeros(par.dim)
            e[j] = 1e-6
            fd = (D.loglik(null_lik, par, s + e, False)[0]
                  - D.loglik(null_lik, par, s - e, False)[0]) / 2e-6  # fmt: skip
            assert fd == pytest.approx(g[j], rel=1e-3, abs=1e-2)


def test_h0_model_equals_config_fixed_b(sim):
    """H0 parametrisation = law=two_type '++law.fixed.B={to_A:0,div_AA:0,div_AB:0}'."""
    from hydra import compose, initialize_config_module

    from bio_sbi.inference.exact_gw import ExactLikelihood
    from bio_sbi.pipeline import build_components

    with initialize_config_module("bio_sbi.conf", version_base=None):
        cfg = compose("config", overrides=["inference=exact_gw",
                                           "++law.fixed.B={to_A:0,div_AA:0,div_AB:0}"])  # fmt: skip
    comp = build_components(cfg)
    assert comp.simulator.space.free_names[-3:] == ("B.death", "B.persist", "B.div_BB")
    data = D.simulate_counts(sim, NULL, HORIZONS, 64, 3, ("reference",))
    lik_cfg = ExactLikelihood(comp.simulator, comp.generator.observation, data)
    lik = D.likelihood(sim, data, kernel="loop")
    P = D.P_from_rows({**A_ROW, "div_AA": 0.5, "div_BB": 0.34},
                      {"death": 0.2, "persist": 0.5, "div_BB": 0.3})  # fmt: skip
    theta = comp.simulator.encode(comp.simulator.physical_from_mapping(D.P_to_dict(P)))
    ours = D.loglik(lik, D.h0_model(), D.h0_model().encode(P), grad=False)[0]
    assert ours == pytest.approx(lik_cfg.log_likelihood(theta, grad=False)[0], abs=1e-4)


def test_lrt_null_fit_and_nonnegative(null_lik):
    r = D.lrt(null_lik, np.random.default_rng(0), FAST)
    assert np.all(r["P0"][1, D.B_DEDIFF] == 0.0)  # H0 fit respects the zeros
    assert r["lambda"] >= 0.0 and r["ll1"] >= r["ll0"] - 1e-9
    assert r["full_ll"]["theta0"] == r["h0_ll"][r["best_start0"]] or \
        r["best_start0"] == "from_full"  # fmt: skip
    # the full model is never worse than its theta_0 candidate
    assert r["ll1"] >= r["full_ll"]["theta0"]


def test_lrt_detects_strong_alternative(sim):
    data = D.simulate_counts(sim, ALT, HORIZONS, 128, 0, ("reference",))
    r = D.lrt(D.likelihood(sim, data), np.random.default_rng(0), FAST)
    assert r["lambda"] > 20 and r["m_ba_hat"] > 0.1


def test_reproducible_with_fixed_seed(sim):
    a = D.simulate_counts(sim, NULL, HORIZONS, 48, 7, ("reference",))
    b = D.simulate_counts(sim, NULL, HORIZONS, 48, 7, ("reference",))
    assert all(np.array_equal(x.counts, y.counts) for x, y in zip(a, b, strict=True))
    r1 = D.run_test(D.likelihood(sim, a), np.random.default_rng(5), FAST, chibar_draws=2000)
    r2 = D.run_test(D.likelihood(sim, b), np.random.default_rng(5), FAST, chibar_draws=2000)
    assert r1["lambda"] == r2["lambda"] and r1["p_chibar"] == r2["p_chibar"]


def test_datasets_nested_and_typing_keeps_clones(sim):
    big = D.simulate_counts(sim, NULL, HORIZONS, 80, 2, ("reference",))
    small = D.subset(big, 30)
    assert all(len(c.counts) == 30 for c in small)
    # perfect typing reproduces the pipeline's reference stream / exact sampler
    conf = D.confusion_matrix(0.3, "b_as_a")
    typed = D.simulate_counts(sim, NULL, HORIZONS, 80, 2, ("reference",), conf)
    for c, t in zip(big, typed, strict=True):
        assert np.array_equal(c.counts.sum(1), t.counts.sum(1))  # same clones, relabelled
        assert np.all(t.counts[:, 0] >= c.counts[:, 0])  # B read as A only


def test_bootstrap_draws_from_theta0(sim, monkeypatch):
    P0 = D.P_from_rows({**A_ROW, "div_AA": 0.6, "div_BB": 0.24},
                       {"death": 0.4, "persist": 0.6})  # fmt: skip
    seen = []
    orig = D.simulate_counts

    def spy(sim_, P, *a, **k):
        seen.append(np.array(P))
        return orig(sim_, P, *a, **k)

    monkeypatch.setattr(D, "simulate_counts", spy)
    out = D.parametric_bootstrap(sim, P0, HORIZONS, 48, range(2), 0, ("t",), settings=FAST)
    assert len(out) == 2 and all(r["lambda"] >= 0 for r in out)
    assert len(seen) == 2 and all(np.array_equal(p, P0) for p in seen)
    # and the replicate data follow theta_0: conditional mean count at the last horizon
    data = D.bootstrap_dataset(sim, P0, (6,), 4000, 0, ("t",), 99)
    phys = D.physical(sim, P0)
    mean = sim.expected_counts(phys, 6) / sim.detection_probability(phys, 6, np.ones(2))
    obs = data[0].counts.mean(0)
    se = data[0].counts.std(0) / np.sqrt(4000)
    assert np.all(np.abs(obs - mean) < 4 * se)
    with pytest.raises(ValueError):
        D.parametric_bootstrap(sim, ALT, HORIZONS, 48, range(1), 0)


def test_chibar_orthogonal_weights_and_pvalues():
    info = np.eye(10)
    loc = D.LocalInfo(info, [f"c{i}" for i in range(10)], [5, 6, 7], [], 1.0)
    cb = D.chibar_null(loc, 40000, seed=0)
    assert np.allclose(cb["weights"], D.BINOMIAL_WEIGHTS, atol=0.01)
    assert cb["crit95"] == pytest.approx(D.chibar_crit(D.BINOMIAL_WEIGHTS), rel=0.03)
    assert D.chibar_pvalue(D.chibar_crit(D.BINOMIAL_WEIGHTS), D.BINOMIAL_WEIGHTS) == \
        pytest.approx(0.05)  # fmt: skip
    assert D.chibar_pvalue(stats.chi2.ppf(0.95, 3), [0, 0, 0, 1]) == pytest.approx(0.05)
    # a boundary nuisance (>= 0 in both cones) leaves Lambda >= 0 and adds no weight
    loc2 = D.LocalInfo(info, loc.names, [5, 6, 7], [2], 1.0)
    cb2 = D.chibar_null(loc2, 20000, seed=0)
    assert np.all(cb2["draws"] >= 0) and cb2["n_boundary_nuisance"] == 1
    assert np.allclose(cb2["weights"], D.BINOMIAL_WEIGHTS, atol=0.015)


def test_local_information_positive_definite_at_null(null_lik):
    r = D.lrt(null_lik, np.random.default_rng(0), FAST)
    loc = D.local_information(null_lik, r["P0"])
    assert loc.info.shape == (10, 10) and np.allclose(loc.info, loc.info.T)
    assert [loc.names[c] for c in loc.dediff] == ["B.to_A", "B.div_AA", "B.div_AB"]


def test_confusion_grid():
    assert np.allclose(D.confusion_grid(0.01, 0.03, 3), [0.01, 0.02, 0.03])
    assert np.allclose(D.confusion_grid(0.02, 0.02, 5), [0.02])
    with pytest.raises(ValueError):
        D.confusion_grid(0.03, 0.01)
    with pytest.raises(ValueError):
        D.confusion_grid(-0.01, 0.01)


def test_robust_combine_takes_most_conservative():
    recs = [
        {"lambda": 9.0, "p_chibar": 0.004, "p_chi2_3": 0.03, "chibar": {"crit95": 3.0}},
        {"lambda": 4.0, "p_chibar": 0.04, "p_chi2_3": 0.26, "chibar": {"crit95": 3.5}},
        {"lambda": 5.0, "p_chibar": 0.06, "p_chi2_3": 0.17, "chibar": {"crit95": 5.5}},
    ]
    out = D.robust_combine(recs, [0.01, 0.015, 0.02], crit=3.8)
    assert out["lambda_min"] == 4.0 and out["rate_lambda_min"] == 0.015
    assert out["p_chibar"] == 0.06 and out["rate_p_chibar"] == 0.02
    assert out["p_chi2_3"] == 0.26 and "p_boot" not in out
    assert out["excess_chibar_min"] == pytest.approx(-0.5)
    assert out["reject_crit"] is True  # smallest Lambda 4.0 > 3.8
    assert D.robust_combine(recs, [0.01, 0.015, 0.02], crit=4.5)["reject_crit"] is False
    with pytest.raises(ValueError):
        D.robust_combine(recs, [0.01, 0.02])


def test_robust_test_runs_each_rate(sim):
    conf = D.confusion_matrix(0.05, "b_as_a")
    data = D.simulate_counts(sim, NULL, HORIZONS, 48, 1, ("reference",), conf)
    rates = D.confusion_grid(0.025, 0.1, 2)
    res = D.robust_test(sim, data, rates, "b_as_a", 3, ("t",), FAST, chibar_draws=2000)
    assert [r["rate"] for r in res["per_rate"]] == list(rates)
    # each rate is the plain test with that confusion matrix and its own starts
    r1 = float(rates[1])
    one = D.run_test(D.likelihood(sim, data, D.confusion_matrix(r1, "b_as_a")),
                     D.rng_for(3, "starts", "t", "rate", r1), FAST, chibar_draws=2000,
                     chibar_seed=D.seed_for(3, "chibar", "t", "rate", r1))  # fmt: skip
    assert one["lambda"] == res["per_rate"][1]["lambda"]
    assert one["p_chibar"] == res["per_rate"][1]["p_chibar"]
    rob = res["robust"]
    assert rob["p_chibar"] == max(r["p_chibar"] for r in res["per_rate"])
    assert rob["lambda_min"] == min(r["lambda"] for r in res["per_rate"])
