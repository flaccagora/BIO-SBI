"""Point estimation for exact_gw (gw_optimise): objectives, PSO, refinement, validation."""

import json

import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import dirichlet

from bio_sbi.inference.exact_gw import ExactGW, ExactLikelihood
from bio_sbi.inference.gw_exact import log_likelihood as raw_log_likelihood
from bio_sbi.inference.gw_optimise import (
    PointObjective,
    lbfgsb,
    minibatch_weights,
    optimise,
    pso,
    settings,
    sgd,
)
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import GaltonWatson
from bio_sbi.summaries import Cohort

from .conftest import load_law

FAST = {"swarm": 8, "max_iter": 8, "restarts": 2, "refine_top": 2, "refine_maxiter": 30,
        "log_every": 0, "validate": False}  # fmt: skip


def make_lik(law_name="reversible_ab", horizons=(3, 4), clones=32, seed=0, **law_kw):
    law = load_law(law_name, **law_kw)
    sim = GaltonWatson(law)
    ref = load_law(law_name)["reference"]
    if "fixed" in law_kw:
        truth = sim.decode(np.full(sim.theta_dim, 0.5))
    else:
        truth = sim.physical_from_mapping(ref)
    rng = np.random.default_rng(seed)
    cohorts = [Cohort(h, sim.simulate_detected(truth, h, clones, np.ones(2), rng))
               for h in horizons]  # fmt: skip
    return ExactLikelihood(sim, ObservationModel(sim.types), cohorts), sim


@pytest.fixture(scope="module")
def rev():
    return make_lik()


def interior(sim, law_name="reversible_ab"):
    """Interior point near the truth (far points floor observed cells: FD meaningless)."""
    truth = sim.physical_from_mapping(load_law(law_name)["reference"])
    return sim.encode(0.8 * truth + 0.2 * sim.decode(np.full(sim.theta_dim, 0.5))).astype(float)


# ---------------------------------------------------------------------- constraints
@pytest.mark.parametrize(
    "law_kw",
    [
        {},
        {
            "fixed": {
                "founder": {"A": 1.0},
                "A": {"death": 0.1},
                "B": {"div_AA": 0.0, "div_AB": 0.0},
            }
        },
    ],
)
def test_decoded_points_are_normalised_and_keep_fixed_entries(law_kw):
    lik, sim = make_lik("two_type", horizons=(3,), clones=16, **law_kw)
    space = sim.space
    rng = np.random.default_rng(0)
    # interior, exact endpoints and mixtures of both
    X = np.concatenate([rng.uniform(size=(20, lik.dim)), rng.integers(0, 2, (20, lik.dim))])
    for x in X:
        phys = space.decode(x)
        space.validate(phys)  # row sums one, nonnegative, fixed entries kept
        assert np.all(phys >= 0)
        start = 0
        for r in space.rows:
            row = phys[start : start + len(r.entries)]
            start += len(r.entries)
            for i, v in r.fixed.items():
                assert row[i] == v
            assert row.sum() == pytest.approx(1.0, abs=1e-14)
        assert np.isfinite(PointObjective(lik).value(x))  # endpoints are evaluable


def test_deterministic_rows_and_fully_fixed_model():
    # B row fully fixed (deterministic): no sticks, no prior term
    fixed_b = {"founder": {"A": 1.0},
               "B": {"death": 0.1, "persist": 0.1, "to_A": 0.8, "div_AA": 0, "div_AB": 0,
                     "div_BB": 0}}  # fmt: skip
    lik, sim = make_lik("two_type", horizons=(3,), clones=16, fixed=fixed_b)
    assert lik.dim == 5 and all(n.startswith("A.") for n in sim.space.theta_names)
    obj = PointObjective(lik, "map")
    assert obj.coef_s.shape == (5,)
    res = optimise(lik, **FAST)
    assert res["best"]["physical"]["B.to_A"] == 0.8
    # everything fixed: the single admissible model is evaluated directly
    ref = load_law("reversible_ab")["reference"]
    all_fixed = {"founder": {"A": 1.0}, **{k: {**{o: 0.0 for o in ("div_AA", "div_AB",
                 "div_BB")}, **v} if k == "B" else v for k, v in ref.items()}}  # fmt: skip
    lik0, sim0 = make_lik("reversible_ab", horizons=(3,), clones=16, fixed=all_fixed)
    assert lik0.dim == 0
    res0 = optimise(lik0, objective="map", validate=True, log_every=0)
    assert res0["value"] == pytest.approx(lik0.log_likelihood(np.zeros(0), grad=False)[0])
    assert res0["best"]["log_prior"] == 0.0 and res0["pso"] == []


# ---------------------------------------------------------------------- likelihood paths
def test_objective_only_path_agrees_with_gradient_path(rev):
    lik, sim = rev
    rng = np.random.default_rng(3)
    n0, g0 = lik.evals, lik.grad_evals
    for x in [*rng.uniform(size=(5, lik.dim)), np.zeros(lik.dim), np.ones(lik.dim)]:
        v, g = lik.log_likelihood(x, grad=False)
        vg, gg = lik.log_likelihood(x, grad=True)
        assert g.shape == (0,) and gg.shape == (lik.dim,)
        assert v == pytest.approx(vg, rel=1e-13, abs=1e-10)
        # gradient path unchanged: same as the Jacobian + kernel call it always made
        P, dP, rows = lik.jacobian(x)
        ref = raw_log_likelihood(lik.model, P, dP, rows, lik.data, lik.floor, lik.min_detection)
        assert vg == ref[0] and np.array_equal(gg, ref[1])
    assert lik.evals - n0 == 14 and lik.grad_evals - g0 == 7


@pytest.mark.parametrize("objective", ["mle", "map"])
def test_objective_gradient_matches_finite_differences(objective):
    conc = {"A": {"death": 2.0, "div_AA": 3.0, "to_B": 1.5}, "B": {"to_A": 4.0}}
    lik, sim = make_lik(concentration=conc)
    obj = PointObjective(lik, objective)
    x = interior(sim)
    v, g = obj.value_and_grad(x)
    assert v == pytest.approx(obj.value(x), rel=1e-13)
    eps = 1e-6
    for i in range(lik.dim):
        e = np.zeros(lik.dim)
        e[i] = eps
        fd = (obj.value(x + e) - obj.value(x - e)) / (2 * eps)
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-4 * np.abs(g).max())


def test_log_prior_is_dirichlet_on_normalised_free_probabilities():
    conc = {"A": {"death": 2.0, "div_AA": 3.0, "to_B": 0.5}, "B": {"to_A": 4.0}}
    lik, sim = make_lik(concentration=conc)
    obj = PointObjective(lik, "map", boundary_eps=1e-6)
    assert obj.name == "map_eps_restricted"
    space = sim.space
    rng = np.random.default_rng(5)
    xs = rng.uniform(0.05, 0.95, size=(4, lik.dim))

    def dirichlet_logpdf(x):
        phys, total, start = space.decode(x), 0.0, 0
        for r in space.rows:
            row = phys[start : start + len(r.entries)]
            start += len(r.entries)
            if r.n_sticks:
                total += dirichlet.logpdf(row[list(r.free)] / r.mass, r.alpha)
        return total

    diffs = [obj.log_prior(x)[0] - dirichlet_logpdf(x) for x in xs]
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-10)  # equal up to a constant
    # gradient of the prior alone
    x, eps = xs[0], 1e-7
    g = obj.log_prior(x)[1]
    for i in range(lik.dim):
        e = np.zeros(lik.dim)
        e[i] = eps
        fd = (obj.log_prior(x + e)[0] - obj.log_prior(x - e)[0]) / (2 * eps)
        assert g[i] == pytest.approx(fd, rel=1e-5, abs=1e-6)


def test_mle_map_and_transformed_posterior_are_distinct_objectives(rev):
    lik, sim = rev
    x = interior(sim)
    mle, mapo = PointObjective(lik, "mle"), PointObjective(lik, "map")
    ll = lik.log_likelihood(x, grad=False)[0]
    # alpha = 1 (default): physical MAP == MLE, but the NUTS density differs (Jacobians)
    assert mle.value(x) == ll and mapo.value(x) == ll

    def nuts_minus_ll(x):  # log_posterior_u - l = stick Beta log density + log Jacobian
        u = np.log(x) - np.log1p(-x)
        th = expit(u)
        rest = lik.log_posterior_u(u)[0] - lik.log_likelihood(x, grad=False)[0]
        stick = beta_dist.logpdf(th, lik.prior_a, lik.prior_b).sum()
        return rest, stick + np.sum(np.log(th) + np.log1p(-th))

    (r1, e1), (r2, e2) = nuts_minus_ll(x), nuts_minus_ll(0.5 * x + 0.25)
    assert r1 - r2 == pytest.approx(e1 - e2, rel=1e-8)  # equal up to Beta constants
    assert abs((r1 - r2)) > 1e-3  # ... and not constant: a different objective
    # alpha != 1: MAP - MLE is the physical log prior only
    lik2, _ = make_lik(concentration={"A": {"death": 2.0}, "B": {"to_A": 3.0}})
    m2 = PointObjective(lik2, "map")
    e = m2.evaluate(x)
    assert e["value"] == pytest.approx(e["log_likelihood"] + m2.log_prior(x)[0])
    assert e["log_prior"] != 0.0
    # alpha < 1: no unrestricted MAP
    lik3, _ = make_lik(concentration={"A": {"death": 0.5}})
    with pytest.raises(ValueError, match="boundary_eps"):
        PointObjective(lik3, "map")
    assert PointObjective(lik3, "mle").name == "mle"  # prior irrelevant for the MLE


# ---------------------------------------------------------------------- PSO
class Stub:
    """Toy objective with the PointObjective interface (NaN on part of the box)."""

    def __init__(self, dim=3, nan_below=None, all_nan=False):
        class L:
            prior_a = np.ones(dim)
            prior_b = np.ones(dim)

        self.lik, self.dim, self.lower, self.upper = L(), dim, 0.0, 1.0
        self.nan_below, self.all_nan = nan_below, all_nan
        self.calls = 0

    def value(self, x):
        self.calls += 1
        if self.all_nan or (self.nan_below is not None and x[0] < self.nan_below):
            return np.nan
        return -float(np.sum((x - 0.3) ** 2))


def test_pso_rejects_invalid_particles_and_fails_without_any():
    s = settings({**FAST, "max_iter": 30, "swarm": 12})
    stub = Stub(nan_below=0.5)
    res = pso(stub, np.random.default_rng(0), s)
    assert res["best_x"][0] >= 0.5 and np.isfinite(res["best_value"])
    assert res["invalid_evals"] > 0 and res["evals"] == stub.calls
    # personal bests are copies: their stored values match re-evaluation
    for x, f in zip(res["pbest_x"], res["pbest_f"], strict=True):
        if np.isfinite(f):
            assert stub.value(x) == f
    assert np.all(np.diff(res["history"]) >= 0)
    with pytest.raises(RuntimeError, match="no initial particle"):
        pso(Stub(all_nan=True), np.random.default_rng(0), s)


@pytest.mark.parametrize("boundary", ["reflect", "clamp"])
def test_pso_stays_in_box_and_reports_stopping(boundary):
    s = settings({**FAST, "max_iter": 200, "swarm": 10, "boundary": boundary, "patience": 10,
                  "max_evals": 500})  # fmt: skip
    res = pso(Stub(), np.random.default_rng(1), s)
    assert np.all((res["pbest_x"] >= 0) & (res["pbest_x"] <= 1))
    assert res["evals"] <= 500 and res["stop_reason"] in ("stagnation", "collapsed", "max_evals")
    assert res["best_value"] > -1e-3


def test_optimise_is_reproducible_and_independent_of_workers(rev):
    lik, _ = rev
    a = optimise(lik, seed=3, workers=1, **FAST)
    b = optimise(lik, seed=3, workers=2, **FAST)
    c = optimise(lik, seed=4, workers=1, **FAST)
    assert a["best"]["theta"] == b["best"]["theta"] and a["value"] == b["value"]
    assert [r["history"] for r in a["pso"]] == [r["history"] for r in b["pso"]]
    assert a["evaluations"] == b["evaluations"]
    assert [r["history"] for r in a["pso"]] != [r["history"] for r in c["pso"]]


def test_refinement_never_returns_a_worse_point():
    class Wrong(Stub):
        def value_and_grad(self, x):
            return self.value(x), 2 * (x - 0.3)  # wrong sign: ascent direction is wrong

    stub = Wrong()
    x0 = np.array([0.6, 0.1, 0.9])
    r = lbfgsb(stub, x0, 50)
    assert r["value"] >= r["start_value"]
    if not r["improved"]:
        assert np.array_equal(r["x"], x0)


def test_pso_refinement_and_multistart_results(rev):
    lik, sim = rev
    res = optimise(lik, seed=0, workers=2, **{**FAST, "validate": True})
    json.dumps(res)  # serialisable
    assert res["objective"] == "mle" and res["evaluations"]["gradient"] > 0
    for c in res["candidates"]:
        assert c["value"] >= c["start_value"]  # refinement kept or improved its start
        assert c["refinement"] is not None
    assert res["value"] == max(c["value"] for c in res["candidates"])
    assert all(v["ok"] for v in res["validation"]["constraints"])
    assert res["validation"]["grids"][0]["max_abs_change"] < 1e-6
    ms = optimise(lik, seed=0, method="multistart", starts=3, refine_maxiter=100, log_every=0,
                  validate=False)  # fmt: skip
    assert len(ms["candidates"]) == 3 and "pso" not in ms
    # both reach the same optimum on this well-identified case
    assert abs(ms["value"] - res["value"]) < 0.5


def test_map_eps_restriction_is_reported():
    lik, sim = make_lik(concentration={"A": {"death": 0.5}})
    res = optimise(lik, objective="map", boundary_eps=1e-4, **{**FAST, "validate": True})
    assert res["objective"] == "map_eps_restricted" and res["box"] == [1e-4, 1 - 1e-4]
    assert "epsilon" in res["validation"]
    assert res["best"]["log_prior"] is not None


def test_exact_gw_point_estimate_does_not_touch_nuts_draws(rev):
    lik, sim = rev
    kw = dict(chains=2, warmup=10, draws=10, init_starts=2, init_maxiter=10, threads=2,
              log_every=0)  # fmt: skip
    cohorts = [Cohort(float(h.generations), np.repeat(np.c_[h.a, h.b], h.weight.astype(int), 0))
               for h in lik.data]  # fmt: skip
    obs = ObservationModel(sim.types)
    plain = ExactGW(**kw).fit_data(sim, obs, cohorts, 4)
    with_opt = ExactGW(**kw, optimise={"enabled": True, **FAST}).fit_data(sim, obs, cohorts, 4)
    assert np.array_equal(plain.draws, with_opt.draws)
    assert plain.optimisation is None and with_opt.optimisation["objective"] == "mle"


def test_pipeline_writes_optimisation_json(tmp_path):
    from bio_sbi.pipeline import run_experiment

    from .test_pipeline import config, save_hydra_config

    cfg = config(
        "inference=exact_gw", "law=reversible_ab", "design.clones=8", "design.horizons=[2,4]",
        "inference.method.log_every=0", "inference.method.chains=2", "inference.method.warmup=10",
        "inference.method.draws=10", "inference.method.init_starts=2",
        "inference.method.init_maxiter=10", "inference.method.threads=2", "inference.sbc_cases=0",
        "budget.ppc_samples=0", "inference.method.optimise.enabled=true",
        "inference.method.optimise.objective=map", "inference.method.optimise.swarm=6",
        "inference.method.optimise.max_iter=5", "inference.method.optimise.log_every=0",
    )  # fmt: skip
    out = tmp_path / "opt"
    save_hydra_config(cfg, out)
    run_experiment(cfg, out)
    saved = json.loads((out / "optimisation.json").read_text())
    assert saved["objective"] == "map" and saved["settings"]["workers"] == 2
    assert "optimisation" not in json.loads((out / "metrics.json").read_text())["exact_gw"]


def test_map_refinement_is_not_stopped_by_zero_density_bounds():
    # alpha > 1: log prior = -inf at some stick bounds; the box keeps a tiny gap there
    lik, sim = make_lik(concentration={"A": {"death": 2.0}, "B": {"to_A": 3.0, "death": 2.0}})
    obj = PointObjective(lik, "map")
    assert np.any(obj.lower > 0) and np.any(obj.upper < 1)
    assert np.isfinite(obj.value(obj.lower)) and np.isfinite(obj.value(obj.upper))
    res = optimise(lik, objective="map", method="multistart", starts=3, refine_maxiter=200,
                   log_every=0, validate=False)  # fmt: skip
    for c in res["candidates"]:
        assert c["refinement"]["nit"] > 3 and c["value"] > c["start_value"]


# ---------------------------------------------------------------------- SGD family
SGD_FAST = {"method": "sgd", "starts": 3, "sgd_iter": 40, "refine_top": 2, "refine_maxiter": 30,
            "log_every": 0, "validate": False}  # fmt: skip


class Quadratic:
    """Toy objective (maximum at ``target``, may lie outside the box) with gradients."""

    def __init__(self, target, sign=1.0, lower=0.0, upper=1.0):
        self.target, self.sign = np.asarray(target, float), sign
        self.dim = len(self.target)
        self.lower, self.upper = np.full(self.dim, lower), np.full(self.dim, upper)
        self.n_clones, self.seen = 1, []

    def value(self, x):
        assert np.all(x >= self.lower) and np.all(x <= self.upper)  # projection holds
        self.seen.append(-float(np.sum((x - self.target) ** 2)))
        return self.seen[-1]

    def value_and_grad(self, x, weights=None):
        return self.value(x), self.sign * -2 * (x - self.target)


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
@pytest.mark.parametrize("coords", ["box", "logit"])
def test_sgd_ascends_the_likelihood_on_a_small_case(rev, optimizer, coords):
    lik, sim = rev
    obj = PointObjective(lik)
    x0 = np.clip(interior(sim) + 0.1, 0, 1)
    lr = {"adam": 0.01, "momentum": 0.002}[optimizer] * (4 if coords == "logit" else 1)
    s = settings({"sgd_optimizer": optimizer, "sgd_coords": coords, "sgd_lr": lr,
                  "sgd_iter": 30, "sgd_patience": 0, "log_every": 0})  # fmt: skip
    res = sgd(obj, x0, s)
    assert res["start_value"] == pytest.approx(obj.value(x0))
    assert res["best_value"] > res["start_value"] + 1.0
    assert res["best_value"] == pytest.approx(obj.value(res["best_x"]))
    # one small step along the gradient increases the objective (direction is ascent)
    one = sgd(obj, x0, {**s, "sgd_iter": 1, "sgd_lr": 1e-4})
    assert one["final_value"] > one["start_value"]


@pytest.mark.parametrize("coords", ["box", "logit"])
def test_sgd_projection_keeps_iterates_in_the_box(coords):
    # maximum outside the box in some coordinates: iterates are projected onto it
    obj = Quadratic([1.5, -0.3, 0.4], lower=0.1, upper=0.9)
    s = settings({"sgd_coords": coords, "sgd_lr": 0.3, "sgd_iter": 200, "sgd_patience": 0,
                  "sgd_schedule": "constant"})  # fmt: skip
    res = sgd(obj, np.array([0.5, 0.5, 0.5]), s)  # Quadratic.value asserts the box
    assert np.all(res["best_x"] >= 0.1) and np.all(res["best_x"] <= 0.9)
    tol = 1e-6 if coords == "box" else 1e-3  # logit: the bound is only approached
    assert np.allclose(res["best_x"], [0.9, 0.1, 0.4], atol=tol)


def test_sgd_on_likelihood_keeps_probabilities_valid_and_fixed_entries():
    fixed = {"founder": {"A": 1.0}, "A": {"death": 0.1}, "B": {"div_AA": 0.0, "div_AB": 0.0}}
    lik, sim = make_lik("two_type", horizons=(3,), clones=16, fixed=fixed)
    for coords in ("box", "logit"):
        res = optimise(lik, sgd_coords=coords, sgd_lr=0.2, sgd_schedule="constant",
                       sgd_polish=False, **SGD_FAST)  # fmt: skip
        for c in res["candidates"]:
            phys = np.array([c["physical"][n] for n in sim.space.names])
            sim.space.validate(phys)
            assert c["physical"]["A.death"] == 0.1 and c["physical"]["B.div_AA"] == 0.0
            theta = np.array([c["theta"][n] for n in sim.space.theta_names])
            assert np.all((theta >= 0) & (theta <= 1))


def test_minibatch_gradient_is_unbiased(rev):
    lik, sim = rev
    obj = PointObjective(lik)
    x = interior(sim)
    v, g = obj.value_and_grad(x)
    rng = np.random.default_rng(0)
    vals, grads = [], []
    for _ in range(400):
        w = minibatch_weights(lik, rng, 0.25)
        for h, wh in zip(lik.data, w, strict=True):
            assert wh.sum() == pytest.approx(h.clones) and len(wh) == len(h.weight)
        vb, gb = obj.value_and_grad(x, weights=w)
        vals.append(vb)
        grads.append(gb)
    G = np.array(grads)
    se = G.std(0) / np.sqrt(len(G))
    assert np.all(np.abs(G.mean(0) - g) < 5 * se + 1e-9)
    assert abs(np.mean(vals) - v) < 5 * np.std(vals) / np.sqrt(len(vals))
    assert np.std(vals) > 0  # really stochastic
    # the full weights reproduce the full-data evaluation exactly
    assert obj.value_and_grad(x, weights=[h.weight for h in lik.data])[0] == v


def test_sgd_is_reproducible_and_uses_the_multistart_starts(rev):
    lik, _ = rev
    kw = {**SGD_FAST, "sgd_batch": 0.5, "sgd_eval_every": 5}
    a = optimise(lik, seed=3, workers=1, **kw)
    b = optimise(lik, seed=3, workers=2, **kw)
    c = optimise(lik, seed=4, workers=1, **kw)
    json.dumps(a)
    assert a["best"]["theta"] == b["best"]["theta"] and a["value"] == b["value"]
    assert [r["history"] for r in a["sgd"]] == [r["history"] for r in b["sgd"]]
    assert a["evaluations"] == b["evaluations"]
    assert [r["history"] for r in a["sgd"]] != [r["history"] for r in c["sgd"]]
    # fair comparison: SGD start i is multistart start i
    ms = optimise(lik, seed=3, method="multistart", starts=3, refine_maxiter=1, log_every=0,
                  validate=False)  # fmt: skip
    assert sorted(r["start_value"] for r in a["sgd"]) == pytest.approx(
        sorted(c["start_value"] for c in ms["candidates"])
    )


def test_sgd_never_returns_a_worse_point_than_its_best_visited():
    # wrong-sign gradient: every step goes downhill; the start must be returned
    obj = Quadratic([0.3, 0.3], sign=-1.0)
    x0 = np.array([0.4, 0.2])
    res = sgd(obj, x0, settings({"sgd_iter": 20, "sgd_lr": 0.05, "sgd_patience": 0}))
    assert np.array_equal(res["best_x"], x0) and res["best_value"] == max(obj.seen)
    assert res["final_value"] < res["best_value"]
    assert np.all(np.diff(res["history"]) >= 0)
    # with polish, a candidate is never worse than its SGD best either
    lik, _ = make_lik(horizons=(3,), clones=16)
    out = optimise(lik, **SGD_FAST)
    for c in out["candidates"]:
        assert c["value"] >= c["start_value"]
    assert out["value"] >= out["sgd_best_value"]
    assert out["evaluations"]["gradient"] >= out["evaluations_before_polish"]["gradient"]


def test_sgd_nonfinite_step_returns_to_best_and_halves_the_rate():
    class Holes(Quadratic):
        def value_and_grad(self, x, weights=None):
            v, g = super().value_and_grad(x)
            return (np.nan, g) if x[0] > 0.45 else (v, g)

    obj = Holes([0.9, 0.5])
    s = settings({"sgd_iter": 50, "sgd_lr": 0.1, "sgd_patience": 0})
    res = sgd(obj, np.array([0.2, 0.5]), s)
    assert res["invalid_steps"] > 0 and res["best_x"][0] <= 0.45
    assert np.isfinite(res["best_value"])


def test_sgd_mle_and_map_are_separate_objectives():
    lik, sim = make_lik(concentration={"A": {"death": 2.0}, "B": {"to_A": 3.0, "death": 2.0}})
    mle = optimise(lik, objective="mle", **SGD_FAST)
    mp = optimise(lik, objective="map", **SGD_FAST)
    assert mle["objective"] == "mle" and mle["best"]["log_prior"] is None
    assert mle["value"] == pytest.approx(mle["best"]["log_likelihood"])
    assert mp["objective"] == "map" and mp["best"]["log_prior"] is not None
    assert mp["value"] == pytest.approx(mp["best"]["log_likelihood"] + mp["best"]["log_prior"])
    # MAP keeps off the zero-density stick bounds, like PSO / L-BFGS-B
    assert mp["bounds"]["lower"] == PointObjective(lik, "map").lower.tolist()


def test_sgd_settings_are_validated():
    for bad in ({"sgd_optimizer": "rmsprop"}, {"sgd_coords": "simplex"},
                {"sgd_schedule": "step"}, {"sgd_unknown": 1}):  # fmt: skip
        with pytest.raises(ValueError):
            settings({"method": "sgd", **bad})
    with pytest.raises(ValueError, match="sgd_batch"):
        sgd(Quadratic([0.5]), np.array([0.2]), settings({"sgd_batch": 1.5}))
