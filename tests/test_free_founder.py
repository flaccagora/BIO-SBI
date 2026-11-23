"""Free founder distribution in inference=exact_gw: gradients, values, progenitor posterior."""

import itertools

import numpy as np
import pytest

from bio_sbi.inference.exact_gw import ExactGW, ExactLikelihood
from bio_sbi.inference.gw_exact import log_likelihood, log_likelihood_founder
from bio_sbi.observation import ObservationModel
from bio_sbi.simulators import GaltonWatson
from bio_sbi.summaries import Cohort

from .conftest import load_law

REF = {
    "A": {"death": 0.04, "persist": 0.07, "to_B": 0.11, "div_AA": 0.42, "div_AB": 0.15,
          "div_BB": 0.21},
    "B": {"death": 0.13, "persist": 0.05, "to_A": 0.36, "div_AA": 0.19, "div_AB": 0.08,
          "div_BB": 0.19},
}  # fmt: skip


def free_sim(sizes=(1, 2)):
    return GaltonWatson(load_law(fixed={}, founder_sizes=list(sizes)))


def mixed_truth(sim, founder_a=0.7, doublet=0.15):
    m = {**REF, "founder": {"A": founder_a, "B": 1 - founder_a}}
    if len(sim.law.founder_sizes) > 1:
        m["founder_size"] = {"1": 1 - doublet, "2": doublet}
    return sim.physical_from_mapping(m)


@pytest.fixture
def data(rng):
    # single founders (doublet-sized clones are numerically impossible under sizes [1]:
    # floored cells make finite differences meaningless)
    sim = free_sim((1,))
    truth = mixed_truth(sim)
    return [Cohort(h, sim.simulate_detected(truth, h, 64, np.ones(2), rng)) for h in (3, 5)]


@pytest.mark.parametrize("kernel", ["loop", "adjoint"])
@pytest.mark.parametrize("sizes", [(1,), (1, 2)])
def test_free_founder_gradient_matches_finite_differences(data, kernel, sizes):
    sim = free_sim(sizes)
    lik = ExactLikelihood(sim, ObservationModel(sim.types), data, kernel=kernel)
    assert lik.founder_free and lik.dim == sim.theta_dim == 10 + len(sizes)
    assert lik.n_row_dims == 10
    theta = sim.encode(mixed_truth(sim, 0.6, 0.2)).astype(float)
    theta = 0.8 * theta + 0.1  # interior
    ll, g = lik.log_likelihood(theta)
    assert np.isfinite(ll) and g.shape == (lik.dim,)
    assert lik.log_likelihood(theta, grad=False)[0] == pytest.approx(ll, rel=1e-12)
    eps = 1e-5
    for i in range(lik.dim):
        e = np.zeros_like(theta)
        e[i] = eps
        fd = (lik.log_likelihood(theta + e, False)[0]
              - lik.log_likelihood(theta - e, False)[0]) / (2 * eps)  # fmt: skip
        assert g[i] == pytest.approx(fd, rel=1e-4, abs=1e-4 * np.abs(g).max())


def test_free_founder_value_equals_fixed_founder_model(data):
    """At the same founder distribution the free and the fixed model agree (both kernels)."""
    free = free_sim()
    fixed = GaltonWatson(load_law(fixed={"founder": {"A": 0.6}, "founder_size": {"2": 0.2}},
                                  founder_sizes=[1, 2]))  # fmt: skip
    obs = ObservationModel(free.types)
    theta_free = free.encode(mixed_truth(free, 0.6, 0.2)).astype(float)
    theta_fixed = theta_free[:10]
    for kernel in ("loop", "adjoint"):
        lf = ExactLikelihood(free, obs, data, kernel=kernel)
        lx = ExactLikelihood(fixed, obs, data, kernel=kernel)
        assert not lx.founder_free
        llf, gf = lf.log_likelihood(theta_free)
        llx, gx = lx.log_likelihood(theta_fixed)
        assert llf == pytest.approx(llx, rel=1e-8)  # round-off (FFT, damping)
        np.testing.assert_allclose(gf[:10], gx, rtol=1e-7, atol=1e-7 * np.abs(gx).max())


def test_fixed_founder_path_is_unchanged(data):
    """Default (fixed A founder): the likelihood still calls the original kernel, bit for bit."""
    sim = GaltonWatson(load_law())
    lik = ExactLikelihood(sim, ObservationModel(sim.types), data, kernel="adjoint")
    assert not lik.founder_free and lik.n_row_dims == lik.dim == sim.theta_dim
    theta = 0.8 * sim.encode(sim.physical_from_mapping(REF)).astype(float) + 0.1
    P, dP, rows = lik.jacobian(theta)
    ref = log_likelihood(lik.model, P, dP, rows, lik.data, lik.floor, lik.min_detection, "adjoint")
    ll, g = lik.log_likelihood(theta)
    assert ll == ref[0] and np.array_equal(g, ref[1])
    # the founder-gradient function agrees with the original one at the fixed founder
    ll2, g2, g_phi, g_pi = log_likelihood_founder(lik.model, P, dP, rows, lik.data, lik.floor,
                                                  lik.min_detection, "adjoint")  # fmt: skip
    assert ll2 == pytest.approx(ll, rel=1e-9)  # round-off (complex powers)
    np.testing.assert_allclose(g2, g, rtol=1e-8, atol=1e-8 * np.abs(g).max())
    assert g_phi.shape == (2,) and g_pi.shape == (1,)


def test_reparam_with_free_founder_raises(data):
    sim = free_sim()
    lik = ExactLikelihood(sim, ObservationModel(sim.types), data, kernel="adjoint")
    with pytest.raises(ValueError, match="founder"):
        ExactGW(reparam="brow").coordinates(lik)


def _brute_force_posterior(P, offspring, phi, pis, counts):
    """One generation: enumerate founders and their outcomes."""
    out = {}
    for m, pi in zip((1, 2), pis, strict=True):
        for types in itertools.product((0, 1), repeat=m):
            label = "".join(sorted("AB"[t] for t in types))
            prior = pi * np.prod([phi[t] for t in types])
            like = 0.0
            for outs in itertools.product(range(6), repeat=m):
                n = sum(offspring[t, o] for t, o in zip(types, outs, strict=True))
                if tuple(n) == tuple(counts):
                    like += np.prod([P[t, o] for t, o in zip(types, outs, strict=True)])
            out[label] = out.get(label, 0.0) + prior * like
    tot = sum(out.values())
    return {k: v / tot for k, v in out.items()}


def test_progenitor_posterior_toy_case():
    sim = free_sim()
    rows = {"A": {"div_AA": 0.5, "persist": 0.3, "to_B": 0.2},
            "B": {"to_A": 0.4, "persist": 0.4, "div_BB": 0.2},
            "founder": {"A": 0.7, "B": 0.3}, "founder_size": {"1": 0.8, "2": 0.2}}  # fmt: skip
    phys = sim.physical_from_mapping(rows)
    counts = np.array([[1, 0], [2, 0], [0, 1], [2, 1], [3, 0], [1, 1], [0, 2]])
    lik = ExactLikelihood(sim, ObservationModel(sim.types), [Cohort(1.0, counts)],
                          kernel="adjoint")  # fmt: skip
    theta = sim.encode(phys).astype(float)
    labels, probs = lik.progenitor_posterior(theta)
    assert labels == ["A", "B", "AA", "AB", "BB"]
    P, model = lik.probabilities(theta), lik.model_at(theta)  # (float32 theta)
    h = lik.data[0]
    for i, (a, b) in enumerate(zip(h.a, h.b, strict=True)):
        bf = _brute_force_posterior(P, sim.law.offspring, model.founder, model.size_probs,
                                    (a, b))  # fmt: skip
        for j, lab in enumerate(labels):
            assert probs[0][i, j] == pytest.approx(bf.get(lab, 0.0), abs=1e-9)
    # single founders: P(A | counts) = p_A L_A / (p_A L_A + p_B L_B)
    single = free_sim((1,))
    rows1 = {k: v for k, v in rows.items() if k != "founder_size"}
    lik1 = ExactLikelihood(single, ObservationModel(single.types), [Cohort(1.0, counts)],
                           kernel="loop")  # fmt: skip
    theta1 = single.encode(single.physical_from_mapping(rows1)).astype(float)
    labels1, probs1 = lik1.progenitor_posterior(theta1)
    assert labels1 == ["A", "B"]
    h1 = lik1.data[0]
    cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(h1.a, h1.b, strict=True))}
    # (1, 0): A persists (0.3) vs B -> A (0.4)
    expect = 0.7 * 0.3 / (0.7 * 0.3 + 0.3 * 0.4)
    assert probs1[0][cell[(1, 0)], 0] == pytest.approx(expect, rel=1e-6)
    assert probs1[0][cell[(0, 2)], 0] == pytest.approx(0.0, abs=1e-12)
