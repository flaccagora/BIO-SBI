from pathlib import Path

import numpy as np
import pytest
import yaml

CONF = Path(__file__).resolve().parents[1] / "src" / "bio_sbi" / "conf"


def load_law(name="reversible_ab"):
    return yaml.safe_load((CONF / "law" / f"{name}.yaml").read_text())


@pytest.fixture
def law():
    return load_law()


@pytest.fixture
def gw(law):
    from bio_sbi.simulators import GaltonWatson

    return GaltonWatson(law["types"], law["outcomes"], law["founder"])


@pytest.fixture
def gw_truth(gw, law):
    return gw.physical_from_mapping(law["reference"])


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
