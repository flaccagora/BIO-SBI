"""BIO-SBI: modular simulation-based inference for cell branching processes.

The package is organised as a pipeline of swappable components, each selected by a
Hydra config group:

* :mod:`bio_sbi.simulators` - generative branching models (Galton-Watson, Bellman-Harris).
* :mod:`bio_sbi.observation` - measurement model (capture, type confusion, clone selection).
* :mod:`bio_sbi.design` - experimental design (endpoints and clones per endpoint).
* :mod:`bio_sbi.summaries` - fixed-length summary statistics of a dataset.
* :mod:`bio_sbi.inference` - sbi inference methods (NPE, NLE, NRE).
* :mod:`bio_sbi.evaluation` - parameter recovery and calibration diagnostics.
* :mod:`bio_sbi.pipeline` - glue that runs one reproducible experiment.
"""

from omegaconf import OmegaConf

__version__ = "0.2.0"

# Config resolver: ``${complement:${..q}}`` = 1 - q (e.g. observation/three_split.yaml).
if not OmegaConf.has_resolver("complement"):
    OmegaConf.register_new_resolver("complement", lambda x: 1.0 - float(x))
