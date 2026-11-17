"""GPU prototypes of the exact GW likelihood (scripts/gw_speedup): agreement with the CPU.

Skipped without a CUDA device (torch) / numba.cuda.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gw_speedup"))

from bio_sbi.inference.exact_gw import ExactLikelihood  # noqa: E402
from bio_sbi.observation import ObservationModel  # noqa: E402
from bio_sbi.simulators import GaltonWatson  # noqa: E402
from bio_sbi.summaries import Cohort  # noqa: E402

from .conftest import load_law  # noqa: E402


def _numba_gw():
    from numba import cuda

    if not cuda.is_available():
        pytest.skip("numba.cuda unavailable")
    from gpu_numba import NumbaGW

    return NumbaGW


@pytest.mark.parametrize("impl", ["torch", "numba"])
@pytest.mark.parametrize("obs_kw", [{}, {"capture": [0.8, 0.9], "selection": "all"}])
def test_gpu_batch_matches_cpu(impl, obs_kw):
    from scipy.special import expit, logit

    spec = load_law("reversible_ab")
    sim = GaltonWatson(spec)
    truth = sim.physical_from_mapping(spec["reference"])
    obs = ObservationModel(sim.types, **obs_kw)
    rng = np.random.default_rng(3)
    cohorts = []
    for h in (3, 5):
        counts = obs.measure(sim.simulate(truth, h, 200, rng), rng)
        if obs.selection == "detected":
            counts = counts[counts.sum(1) > 0]
        cohorts.append(Cohort(h, counts[:64]))
    lik = ExactLikelihood(sim, obs, cohorts)
    if impl == "torch":
        from gpu_torch import TorchGW as Impl
    else:
        Impl = _numba_gw()
    gpu = Impl(lik)
    th = sim.encode(truth).astype(float)
    pts = expit(logit(np.clip(th, 1e-6, 1 - 1e-6))[None] + rng.normal(0, 0.15, (5, len(th))))
    ll, g = gpu.value_and_grad(pts)
    ll0, _ = gpu.value_and_grad(pts, grad=False)
    for i, t in enumerate(pts):
        v, gc = lik.log_likelihood(t)
        assert ll[i] == pytest.approx(v, rel=1e-10)
        assert ll0[i] == pytest.approx(v, rel=1e-10)
        np.testing.assert_allclose(g[i], gc, rtol=1e-8, atol=1e-8 * np.abs(gc).max())
