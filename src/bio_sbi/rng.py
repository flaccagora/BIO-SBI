"""Keyed, order-independent random streams.

Every stochastic step draws from its own generator derived from ``(root seed, *keys)``.
Results therefore do not depend on execution order, thread scheduling or how many
other datasets were simulated, which makes parallel and resumed runs reproducible.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_for(root: int, *keys: object) -> int:
    """Derive a 32-bit seed from a root seed and an arbitrary tuple of keys.

    Args:
        root: Experiment-level seed.
        *keys: Any objects with a stable ``str`` representation (e.g. ``"train", 3``).

    Returns:
        An integer in ``[0, 2**32)``.
    """
    payload = "|".join([str(root), *(str(k) for k in keys)]).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**32)


def rng_for(root: int, *keys: object) -> np.random.Generator:
    """Return an independent NumPy generator for ``(root, *keys)``."""
    return np.random.default_rng(seed_for(root, *keys))


def seed_torch(seed: int, deterministic: bool = True) -> None:
    """Seed Python and PyTorch (CPU and CUDA) global generators.

    sbi draws prior samples, training splits and MCMC initialisations from the global
    torch generator, so it is re-seeded before each stochastic torch stage.

    Args:
        seed: Seed value.
        deterministic: Request deterministic torch kernels where available.
    """
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
