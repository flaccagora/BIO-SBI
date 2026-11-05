"""Minimal, dependency-light diagnostic figures (Matplotlib, headless backend)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _grid(n: int) -> tuple[plt.Figure, np.ndarray]:
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def posterior_marginals(
    samples: np.ndarray,
    prior_samples: np.ndarray,
    truth: np.ndarray | None,
    names: Sequence[str],
    path: str | Path,
) -> None:
    """Histogram of each posterior marginal over the prior, with the true value."""
    fig, axes = _grid(len(names))
    for i, (ax, name) in enumerate(zip(axes, names, strict=False)):
        bins = np.linspace(
            min(prior_samples[:, i].min(), samples[:, i].min()),
            max(prior_samples[:, i].max(), samples[:, i].max()),
            40,
        )
        ax.hist(prior_samples[:, i], bins=bins, density=True, color="0.8", label="prior")
        ax.hist(samples[:, i], bins=bins, density=True, alpha=0.75, color="C0", label="posterior")
        if truth is not None:
            ax.axvline(truth[i], color="C3", lw=1.5, label="truth")
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sbc_ranks(ranks: np.ndarray, posterior_samples: int, names: Sequence[str], path) -> None:
    """Rank histograms with the 99% band expected under perfect calibration."""
    from scipy.stats import binom

    fig, axes = _grid(len(names))
    n_bins = int(min(20, max(5, len(ranks) // 10)))
    lo, hi = binom.ppf([0.005, 0.995], len(ranks), 1 / n_bins)
    for i, (ax, name) in enumerate(zip(axes, names, strict=False)):
        ax.axhspan(lo, hi, color="0.85")
        ax.hist(ranks[:, i], bins=n_bins, range=(0, posterior_samples), color="C0")
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
