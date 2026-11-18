"""exact_gw sampler options: block-wise NUTS, warm start, ESS / R-hat stopping rule."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from bio_sbi.inference import ExactGW
from bio_sbi.inference.exact_gw import (
    STOP_DEFAULTS,
    _to_u,
    _within_covariance,
    load_previous_posterior,
    spread_starts,
)
from bio_sbi.inference.nuts import NutsChain, run_chain
from bio_sbi.rng import rng_for
from bio_sbi.simulators import GaltonWatson

from .conftest import load_law


class GaussLik:
    """Stand-in for ExactLikelihood: Gaussian (mixture) log density on ``u``.

    ``modes``: list of means (equal weights, common covariance). ``log_posterior_u``,
    ``space``, ``dim``, ``prior_a/b``, ``data`` are what ``ExactGW.run`` uses.
    """

    def __init__(self, modes, cov, clones=100):
        sim = GaltonWatson(load_law("reversible_ab"))
        self.space = sim.space
        self.dim = self.space.theta_dim
        self.modes = [np.asarray(m, float) for m in modes]
        self.prec = np.linalg.inv(cov)
        self.prior_a = np.ones(self.dim)
        self.prior_b = np.ones(self.dim)
        grid = SimpleNamespace(Lx=8, Ly=8)
        self.data = [SimpleNamespace(clones=clones, generations=4, grid=grid, a=np.zeros(3))]
        self.evals, self.seconds_per_eval = 0, float("nan")

    def log_posterior_u(self, u):
        self.evals += 1
        r = [u - m for m in self.modes]
        lps = np.array([-0.5 * x @ self.prec @ x for x in r])
        top = lps.max()
        w = np.exp(lps - top)
        lp = top + np.log(w.sum() / len(w))
        g = -sum(wi * (self.prec @ x) for wi, x in zip(w, r, strict=True)) / w.sum()
        return float(lp), g


def correlated_cov(d, sd=0.3, rho=0.5):
    c = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    return sd**2 * c


def method(**kw):
    opts = {"chains": 4, "warmup": 100, "draws": 100, "threads": 1, "init_starts": 4,
            "init_maxiter": 50, "log_every": 0, "seed": 3}  # fmt: skip
    return ExactGW(**{**opts, **kw})


# ---------------------------------------------------------------- block-wise NUTS
def test_block_draws_reproduce_a_one_shot_chain():
    cov = correlated_cov(3, 1.0, 0.6)
    prec = np.linalg.inv(cov)

    def f(q):
        return -0.5 * q @ prec @ q, -prec @ q

    ref = run_chain(f, np.ones(3), 120, 90, np.random.default_rng(5))
    chain = NutsChain(f, np.ones(3), np.random.default_rng(5))
    chain.adapt(120)
    for n in (10, 33, 47):
        chain.draw(n)
    res = chain.result()
    np.testing.assert_array_equal(res.draws, ref.draws)
    assert res.step_size == ref.step_size and res.gradient_evals == ref.gradient_evals
    assert 0 < res.warmup_gradient_evals < res.gradient_evals
    assert res.warmup_seconds <= res.seconds


def test_defaults_reproduce_the_plain_run():
    """Features off: the same draws as the original chain loop (run_chain per chain)."""
    lik = GaussLik([np.full(7, 0.2)], correlated_cov(7))
    m = method(chains=2, warmup=60, draws=40)
    draws, diag = m.run(lik)
    # the original implementation: L-BFGS inits, Laplace start, run_chain
    inits, _ = m._initial_points(lik, 4, 2, ("reference",))
    ref = []
    for c in range(2):
        rng = rng_for(m.seed, "exact-nuts", "reference", c)
        cov = m.laplace_covariance(lik, inits[c])
        q0 = inits[c] + np.linalg.cholesky(cov) @ rng.standard_normal(len(cov))
        ref.append(run_chain(lik.log_posterior_u, q0, 60, 40, rng, inv_mass=cov).draws)
    np.testing.assert_array_equal(draws, expit(np.concatenate(ref)))
    assert "warm_start" not in diag and "stop" not in diag
    assert diag["draws_per_chain"] == 40 and diag["warmup"] == 60
    assert diag["warmup_gradient_evals"] + diag["sampling_gradient_evals"] == diag["gradient_evals"]
    assert 0 <= diag["idle_fraction"] < 1
    # stop enabled but never checked (min_draws > max_draws = draws): same draws again
    stop = {**STOP_DEFAULTS, "enabled": True, "block": 15, "min_draws": 99, "max_draws": 40}
    draws2, diag2 = m.run(lik, stop=stop)
    np.testing.assert_array_equal(draws2, draws)
    assert diag2["stop"]["reason"] == "max_draws" and diag2["stop"]["draws_per_chain"] == 40


# ---------------------------------------------------------------- stopping rule
def test_stop_rule_stops_on_an_easy_posterior():
    lik = GaussLik([np.zeros(7)], correlated_cov(7))
    stop = {**STOP_DEFAULTS, "enabled": True, "ess": 300, "tail_ess": 150, "rhat": 1.02,
            "block": 25, "min_draws": 50, "max_draws": 1000}  # fmt: skip
    draws, diag = method().run(lik, stop=stop)
    s = diag["stop"]
    assert s["reason"] == "targets_met" and s["met"] and s["failed"] == []
    assert s["draws_per_chain"] < 1000 and len(draws) == 4 * s["draws_per_chain"]
    last = s["trajectory"][-1]
    assert last["min_ess_bulk"] >= 300 and last["max_rhat"] <= 1.02
    assert last["min_ess_tail"] >= 150 and last["max_mean_spread"] <= 0.5
    # checks only from min_draws on; draws in blocks of 25
    assert [t["draws_per_chain"] for t in s["trajectory"]][:2] == [25, 50]
    assert "max_rhat" not in s["trajectory"][0]
    assert diag["min_ess_bulk"] == pytest.approx(last["min_ess_bulk"])


def _two_mode_chains(lik, cov):
    """Two chains in each of two far-apart modes (NUTS never crosses between them)."""
    chains = []
    for c in range(4):
        mode = lik.modes[c % 2]
        ch = NutsChain(lik.log_posterior_u, mode.copy(), np.random.default_rng(c), inv_mass=cov)
        ch.adapt(60)
        chains.append(ch)
    return chains


@pytest.mark.parametrize("rhat", [1.01, 100.0])
def test_stop_rule_does_not_stop_on_chains_in_different_modes(rhat):
    """R-hat catches it; with R-hat switched off the chain-mean spread still does."""
    cov = correlated_cov(7, 0.2, 0.0)
    lik = GaussLik([np.full(7, -1.5), np.full(7, 1.5)], cov)  # 7.5 sd apart per axis
    m = method()
    objs = _two_mode_chains(lik, cov)
    stop = {**STOP_DEFAULTS, "enabled": True, "ess": 100, "tail_ess": 50, "rhat": rhat,
            "block": 50, "min_draws": 50, "max_draws": 200}  # fmt: skip
    with ThreadPoolExecutor(1) as pool:
        info = m._draw_until(objs, pool, 1, lik, stop, "test")
    assert info["reason"] == "max_draws" and not info["met"]
    assert "mean_spread" in info["failed"]
    if rhat < 2:
        assert "rhat" in info["failed"]
    assert all(t["max_mean_spread"] > 1.5 for t in info["trajectory"])


# ---------------------------------------------------------------- warm start
def _previous_run(tmp_path, lik, rng, n_chains=4, n_draws=150, clones=50, modes=None):
    """Fake previous run directory: chain-major draws in two clusters of u."""
    modes = modes if modes is not None else [np.full(lik.dim, -1.0), np.full(lik.dim, 1.0)]
    cov_prev = correlated_cov(lik.dim, 0.4)
    u = np.concatenate([rng.multivariate_normal(modes[c % len(modes)], cov_prev, n_draws)
                        for c in range(n_chains)])  # fmt: skip
    theta = expit(u)
    phys = lik.space.decode(theta)
    out = tmp_path / "prev"
    out.mkdir()
    pd.DataFrame(phys, columns=lik.space.names).to_csv(out / "posterior_samples.csv.gz",
                                                      index=False)  # fmt: skip
    diag = {"chains": n_chains, "draws_per_chain": n_draws, "step_size": [0.3, 0.5, 0.4, 0.6],
            "grids": [{"clones": clones}, {"clones": clones}]}  # fmt: skip
    (out / "metrics.json").write_text(json.dumps({"exact_gw": diag}))
    return out, u


def test_warm_start_uses_previous_draws_and_scaled_covariance(tmp_path):
    rng = np.random.default_rng(0)
    lik = GaussLik([np.zeros(7)], correlated_cov(7, 0.2), clones=200)  # n = 200 vs 100
    out, u = _previous_run(tmp_path, lik, rng)
    prev = load_previous_posterior(out, lik.space)
    assert prev.clones == 100 and prev.chain is not None and len(prev.theta) == 600
    np.testing.assert_allclose(_to_u(prev.theta), u, atol=1e-5)  # csv round trip
    m = method()
    starts, cov, step, warmup, adapt, info = m._warm_plan(lik, prev, 4, ("reference",))
    # starts are previous draws, from both clusters
    for s, i in zip(starts, info["start_indices"], strict=True):
        np.testing.assert_allclose(s, _to_u(prev.theta)[i])
    assert {np.sign(s.mean()) for s in starts} == {-1.0, 1.0}
    # mass matrix: previous within-chain covariance x n_prev / n (not the between-mode spread)
    within = _within_covariance(_to_u(prev.theta), prev.chain)
    np.testing.assert_allclose(cov, 0.5 * within + 1e-10 * np.eye(7), rtol=1e-8)
    assert info["scale"] == 0.5 and np.trace(cov) < 0.5 * np.trace(np.cov(u.T))
    assert step == pytest.approx(0.45) and warmup == 150
    assert adapt["window_ends"] == [100] and adapt["collect_from"] == 25
    # overrides
    m2 = method(warm_start_options={"scale": 0.25, "warmup": 40, "init_buffer": 10,
                                    "term_buffer": 20, "step_size": "heuristic"})  # fmt: skip
    _, cov2, step2, warmup2, adapt2, info2 = m2._warm_plan(lik, prev, 4, ("reference",))
    np.testing.assert_allclose(cov2, 0.25 * within + 1e-10 * np.eye(7), rtol=1e-8)
    assert step2 is None and warmup2 == 40 and adapt2["window_ends"] == [20]
    # full run: no L-BFGS, chains start where planned, short warmup recorded
    draws, diag = m.run(lik, warm=prev)
    w = diag["warm_start"]
    assert w["start_indices"] == info["start_indices"] and diag["warmup"] == 150
    assert diag["init_log_posterior"] == [] and len(draws) == 400
    assert np.all(np.isfinite(draws)) and diag["divergences"] == 0
    assert all(0.6 < a < 0.95 for a in diag["accept_stat"])  # step size adapted


def test_warm_start_from_posterior_pt_and_csv_agree(tmp_path):
    from bio_sbi.inference.exact_gw import ExactPosterior

    rng = np.random.default_rng(1)
    lik = GaussLik([np.zeros(7)], correlated_cov(7, 0.2), clones=200)
    out, u = _previous_run(tmp_path, lik, rng)
    diag = json.loads((out / "metrics.json").read_text())["exact_gw"]
    post = ExactPosterior(posterior=None, feature_mask=np.ones(3, bool), method="exact_gw",
                          amortized=False, draws=expit(u), diagnostics=diag)  # fmt: skip
    post.save(out / "posterior.pt")
    a = load_previous_posterior(out, lik.space)  # directory: posterior.pt first
    b = load_previous_posterior(out / "posterior_samples.csv.gz", lik.space)
    assert a.source.endswith("posterior.pt") and b.source.endswith(".csv.gz")
    np.testing.assert_allclose(a.theta, b.theta, rtol=1e-6, atol=1e-9)
    assert a.clones == b.clones == 100
    np.testing.assert_array_equal(a.chain, b.chain)


def test_warm_start_without_clone_count_needs_n_prev(tmp_path):
    rng = np.random.default_rng(2)
    lik = GaussLik([np.zeros(7)], correlated_cov(7, 0.2), clones=200)
    out, _ = _previous_run(tmp_path, lik, rng)
    (out / "metrics.json").unlink()
    prev = load_previous_posterior(out, lik.space)
    assert prev.chain is None and prev.clones is None
    with pytest.raises(ValueError, match="n_prev"):
        method()._warm_plan(lik, prev, 4, ("reference",))
    _, _, step, _, _, info = method(warm_start_options={"n_prev": 300})._warm_plan(
        lik, prev, 4, ("reference",)
    )
    assert info["scale"] == pytest.approx(1.5) and step is None


def test_spread_starts_cover_separate_modes():
    rng = np.random.default_rng(3)
    u = np.concatenate([rng.normal(-3, 0.3, (900, 4)), rng.normal(3, 0.3, (100, 4))])
    clusters = spread_starts(u, 4, np.random.default_rng(0))
    firsts = u[[c[0] for c in clusters]]
    assert (firsts[:, 0] > 0).sum() >= 1 and (firsts[:, 0] < 0).sum() >= 1


def test_warm_start_and_stop_reach_the_method_from_the_config():
    from bio_sbi.pipeline import build_components

    from .test_pipeline import config

    m = build_components(config("inference=exact_gw")).method
    assert m.warm_start is None and m.stop["enabled"] is False
    assert m.warm_start_options["warmup"] == 150
    cfg = config("inference=exact_gw", "inference.method.warm_start=/tmp/prev_run",
                 "inference.method.warm_start_options.scale=0.5",
                 "inference.method.stop.enabled=true", "inference.method.stop.ess=300",
                 "inference.method.stop.max_draws=800")  # fmt: skip
    m = build_components(cfg).method
    assert m.warm_start == "/tmp/prev_run" and m.warm_start_options["scale"] == 0.5
    assert m.stop["enabled"] and m.stop["ess"] == 300 and m.stop["max_draws"] == 800
    with pytest.raises(ValueError, match="unknown"):
        ExactGW(stop={"esss": 3})


def test_pipeline_warm_start_and_stop(tmp_path):
    """Cold tiny run, then a warm-started run with the stopping rule from its directory."""
    from bio_sbi.pipeline import run_experiment

    from .test_pipeline import config, save_hydra_config

    base = ("inference=exact_gw", "law=reversible_ab", "design.horizons=[2,4]",
            "inference.method.log_every=0", "inference.method.chains=2",
            "inference.method.warmup=30", "inference.method.draws=30",
            "inference.method.init_starts=2", "inference.method.init_maxiter=20",
            "inference.method.threads=1", "inference.sbc_cases=0", "budget.ppc_samples=5")  # fmt: skip
    cold = tmp_path / "cold"
    cfg = config(*base, "design.clones=16")
    save_hydra_config(cfg, cold)
    run_experiment(cfg, cold)
    warm = tmp_path / "warm"
    cfg = config(*base, "design.clones=32", f"inference.method.warm_start={cold}",
                 "inference.method.warm_start_options.warmup=40",
                 "inference.method.stop.enabled=true", "inference.method.stop.block=10",
                 "inference.method.stop.min_draws=20", "inference.method.stop.max_draws=40",
                 "inference.method.stop.ess=20", "inference.method.stop.tail_ess=10",
                 "inference.method.stop.rhat=1.5", "inference.method.stop.max_mean_spread=3")  # fmt: skip
    save_hydra_config(cfg, warm)
    metrics = run_experiment(cfg, warm)
    d = metrics["exact_gw"]
    assert d["warm_start"]["source"].endswith("posterior.pt")
    assert d["warm_start"]["scale"] == pytest.approx(0.5) and d["warmup"] == 40
    assert d["stop"]["reason"] in ("targets_met", "max_draws")
    n_kept = min(int(cfg.budget.posterior_samples), 2 * d["draws_per_chain"])
    assert metrics["posterior_samples_drawn"] == n_kept
    assert json.loads((warm / "metrics.json").read_text())["exact_gw"]["stop"]["trajectory"]
