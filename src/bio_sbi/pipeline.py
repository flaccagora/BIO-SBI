"""Run one reproducible synthetic experiment from a Hydra config.

Stages (each a small function so they can be reused from notebooks):

1. :func:`build_components` - instantiate simulator, observation model, design,
   summary and inference method from config;
2. :func:`reference_parameters` - pick the generating parameters of the reference
   ("pseudo-observed") dataset;
3. :func:`run_experiment` - simulate the reference dataset, train the posterior,
   evaluate recovery and calibration, and write all artefacts to the run directory.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from . import plots
from .evaluation import calibration, recovery_table
from .inference import FittedPosterior, InferenceMethod
from .provenance import write_manifest
from .rng import seed_for, seed_torch
from .simulate import DataGenerator
from .simulators import BranchingSimulator

LOG = logging.getLogger(__name__)


@dataclass
class Components:
    """Everything instantiated from one config."""

    simulator: BranchingSimulator
    generator: DataGenerator
    method: InferenceMethod


def build_components(cfg: DictConfig) -> Components:
    """Instantiate all pipeline components from a composed config."""
    simulator = instantiate(cfg.simulator.model, _convert_="all")
    generator = DataGenerator(
        simulator=simulator,
        observation=instantiate(cfg.observation, _convert_="all"),
        design=instantiate(cfg.design, _convert_="all"),
        summary=instantiate(cfg.summary, _convert_="all"),
        seed=int(cfg.seed),
        workers=int(cfg.workers),
    )
    method = instantiate(cfg.inference.method, _convert_="all", device=cfg.device, seed=cfg.seed)
    return Components(simulator, generator, method)


def reference_parameters(cfg: DictConfig, simulator: BranchingSimulator) -> np.ndarray:
    """Physical parameters of the reference dataset.

    ``truth.source=reference`` merges ``law.reference``, ``simulator.reference`` and
    ``truth.values`` (later wins); ``truth.source=prior`` draws one prior sample.
    """
    if cfg.truth.source == "prior":
        seed_torch(seed_for(cfg.seed, "truth"))
        theta = simulator.prior().sample((1,)).numpy()[0]
        return simulator.decode(theta)
    if cfg.truth.source != "reference":
        raise ValueError("truth.source must be 'reference' or 'prior'")
    values: dict = {}
    for section in (
        cfg.law.get("reference"),
        cfg.simulator.get("reference"),
        cfg.truth.get("values"),
    ):
        if section is not None:
            values.update(OmegaConf.to_container(section))
    return simulator.physical_from_mapping(values)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float))


def run_experiment(cfg: DictConfig, out: str | Path) -> dict:
    """Simulate, train, evaluate and save one experiment into ``out``.

    Returns:
        The metrics dictionary also written to ``out/metrics.json``.
    """
    out = Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    write_manifest(out)
    comp = build_components(cfg)
    sim, gen = comp.simulator, comp.generator
    prior = sim.prior()

    # 1. Reference ("observed") dataset with known truth.
    truth = reference_parameters(cfg, sim)
    reference = gen.dataset(truth, "reference")
    if reference is None:
        raise RuntimeError("Reference parameters exceed simulation limits; adjust truth/limits")
    x_obs = gen.summarize(reference)
    np.savez_compressed(
        out / "reference.npz",
        truth=truth,
        x=x_obs,
        **{f"counts_h{c.horizon:g}": c.counts for c in reference},
    )
    LOG.info("Reference clones: %s", {c.horizon: c.counts.sum(0).tolist() for c in reference})

    # 2. Train.
    kept: list[tuple[np.ndarray, np.ndarray]] = []

    def simulate(theta: np.ndarray, r: int) -> np.ndarray:
        x, _ = gen.simulate_theta(theta, f"round-{r}")
        LOG.info("Round %d: %d/%d valid simulations", r, np.isfinite(x).all(1).sum(), len(x))
        if cfg.save_training_data:
            kept.append((theta, x))
        return x

    fitted = comp.method.fit(
        prior, simulate, int(cfg.budget.simulations), x_obs, log_dir=out / "tensorboard"
    )
    fitted.save(out / "posterior.pt")
    if kept:
        np.savez_compressed(
            out / "training_data.npz",
            theta=np.concatenate([t for t, _ in kept]),
            x=np.concatenate([x for _, x in kept]),
        )

    # 3. Recovery at the reference truth.
    t_sample = time.perf_counter()
    samples_theta = fitted.sample(
        x_obs, int(cfg.budget.posterior_samples), seed_for(cfg.seed, "ref")
    )
    sample_seconds = time.perf_counter() - t_sample
    samples = sim.decode(samples_theta)
    seed_torch(seed_for(cfg.seed, "prior-reference"))
    prior_samples = sim.decode(prior.sample((4096,)).numpy())
    rows = recovery_table(samples, truth, prior_samples, sim.parameter_names)
    pd.DataFrame(rows).to_csv(out / "recovery.csv", index=False)
    pd.DataFrame(samples, columns=sim.parameter_names).to_csv(
        out / "posterior_samples.csv.gz", index=False
    )
    plots.posterior_marginals(
        samples, prior_samples, truth, sim.parameter_names, out / "figures" / "posterior.png"
    )

    metrics: dict = {
        "simulator": cfg.simulator.name,
        "law": cfg.law.name,
        "inference": cfg.inference.name,
        "rounds": comp.method.rounds,
        "simulations": int(cfg.budget.simulations),
        "design": {"horizons": list(gen.design.horizons), "clones": list(gen.design.clones)},
        "x_dim": int(fitted.feature_mask.sum()),
        "training": fitted.history,
        "sampling_seconds": sample_seconds,
        "recovery": {
            "mean_abs_error": float(np.mean([r["abs_error"] for r in rows])),
            "mean_contraction": float(np.nanmean([r["contraction"] for r in rows])),
            "coverage90": float(np.mean([r["covered"] for r in rows])),
            "per_parameter": rows,
        },
    }

    # 4. Calibration (amortised posteriors only).
    cases = int(cfg.budget.sbc_cases)
    if cases > 0 and fitted.amortized:
        seed_torch(seed_for(cfg.seed, "sbc-theta"))
        theta_sbc = prior.sample((cases,)).numpy()
        x_sbc, _ = gen.simulate_theta(theta_sbc, "sbc")
        cal, ranks = calibration(
            fitted, theta_sbc, x_sbc, sim.theta_names, int(cfg.budget.sbc_samples), cfg.seed
        )
        metrics["calibration"] = cal
        np.savez_compressed(out / "sbc.npz", ranks=ranks, theta=theta_sbc, x=x_sbc)
        plots.sbc_ranks(
            ranks, int(cfg.budget.sbc_samples), sim.theta_names, out / "figures" / "sbc.png"
        )
    elif cases > 0:
        metrics["calibration"] = "skipped: sequential posterior is specific to the reference"

    metrics["total_seconds"] = time.perf_counter() - t0
    _write_json(out / "metrics.json", metrics)
    LOG.info(
        "Done in %.1fs: MAE=%.4f contraction=%.3f -> %s",
        metrics["total_seconds"],
        metrics["recovery"]["mean_abs_error"],
        metrics["recovery"]["mean_contraction"],
        out,
    )
    return metrics


def load_run(run_dir: str | Path) -> tuple[DictConfig, Components, FittedPosterior]:
    """Reload a finished run: its config, components and trained posterior."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return cfg, build_components(cfg), FittedPosterior.load(run_dir / "posterior.pt")


def infer_measured(run_dir: str | Path, data: str | Path, out: str | Path, n: int = 4096) -> Path:
    """Condition a trained posterior on measured counts and save posterior samples.

    Args:
        run_dir: Directory of a finished amortised run.
        data: CSV with one row per clone: a ``horizon`` column plus one count column
            per cell type (named as in the law, e.g. ``A``, ``B``).
        out: Output directory.
        n: Number of posterior samples.

    Returns:
        The output directory.
    """
    from .summaries import Cohort

    cfg, comp, fitted = load_run(run_dir)
    if not fitted.amortized:
        raise ValueError("Measured inference needs an amortised (rounds=1) posterior")
    sim, gen = comp.simulator, comp.generator
    table = pd.read_csv(data)
    missing = {"horizon", *sim.types} - set(table.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {sorted(missing)}")
    cohorts = []
    for horizon, size in gen.design.cohorts:
        rows = table[np.isclose(table["horizon"], horizon)]
        if len(rows) == 0:
            raise ValueError(f"No clones at horizon {horizon:g}; trained design: {gen.design}")
        if len(rows) != size:
            LOG.warning("Horizon %g: %d clones measured, %d trained", horizon, len(rows), size)
        cohorts.append(Cohort(horizon, rows[list(sim.types)].to_numpy(dtype=np.int64)))
    extra = set(table["horizon"]) - set(gen.design.horizons)
    if extra:
        raise ValueError(f"Horizons {sorted(extra)} were not part of the trained design")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = sim.decode(fitted.sample(gen.summarize(cohorts), n, seed_for(cfg.seed, "measured")))
    frame = pd.DataFrame(samples, columns=sim.parameter_names)
    frame.to_csv(out / "posterior_samples.csv.gz", index=False)
    frame.describe(percentiles=[0.05, 0.5, 0.95]).T.to_csv(out / "posterior_summary.csv")
    seed_torch(seed_for(cfg.seed, "prior-reference"))
    prior_samples = sim.decode(sim.prior().sample((4096,)).numpy())
    plots.posterior_marginals(
        samples, prior_samples, None, sim.parameter_names, out / "posterior.png"
    )
    return out
