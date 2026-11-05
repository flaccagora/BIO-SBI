from pathlib import Path

import numpy as np
import pytest
import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "bio_sbi" / "conf"


def pytest_configure(config):
    # Tiny networks gain nothing from many threads, and on a loaded machine torch's
    # default (one thread per core) oversubscribes the CPUs and stalls MCMC sampling.
    import torch

    torch.set_num_threads(2)


def load_law(name="two_type", **overrides):
    spec = yaml.safe_load((CONF / "law" / f"{name}.yaml").read_text())
    spec.update(overrides)
    return spec


@pytest.fixture
def law():
    return load_law()


@pytest.fixture
def gw(law):
    from bio_sbi.simulators import GaltonWatson

    return GaltonWatson(law)


@pytest.fixture
def gw_truth(gw, law):
    return gw.physical_from_mapping(law["reference"])


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
