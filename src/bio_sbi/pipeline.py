"""Run one reproducible synthetic experiment from a Hydra config.

Two models are built from one config:

* the **inference model** (``simulator``, ``law``, ``observation``): what we *assume*;
  used for training simulations, the prior and calibration;
* the **data model** (``data.simulator``, ``data.law``, ``data.observation``): what
  *generates* the reference dataset. It mirrors the inference model by default and
  can be changed independently (different types, transitions, founders, fixed values
  or assay) to test the consequences of a wrong assumption.

Stages (small functions, reusable from notebooks):

1. :func:`build_components` / :func:`build_data_generator` - instantiate both models;
2. :func:`reference_parameters` - generating parameters of the reference dataset;
3. :func:`run_experiment` - simulate the reference data, train, evaluate recovery,
   calibration and posterior-predictive fit, and save all artefacts.
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
from .evaluation import calibration, posterior_predictive_check, recovery_table
from .inference import FittedPosterior, InferenceMethod
from .provenance import write_manifest
from .rng import seed_for, seed_torch
from .simulate import DataGenerator
from .simulators import BranchingSimulator
from .simulators.law import flatten

LOG = logging.getLogger(__name__)


@dataclass
class Components:
    """Inference-model components instantiated from one config."""

    simulator: BranchingSimulator
    generator: DataGenerator
    method: InferenceMethod


def data_seed(cfg: DictConfig) -> int:
    """Root seed of the data side: the truth draw and the reference dataset.

    ``data_seed`` defaults to ``seed`` (configs written before it existed lack the key).
    Everything on the inference side (training simulations, network training, posterior
    sampling, prior samples for contraction, PPC, SBC) keeps using ``seed``.
    """
    value = cfg.get("data_seed")
    return int(cfg.seed if value is None else value)


def _generator(cfg: DictConfig, section: DictConfig, seed: int) -> DataGenerator:
    """Build a data generator from a ``{simulator, observation}`` config section."""
    simulator = instantiate(section.simulator.model, _convert_="all")
    observation = instantiate(section.observation, _convert_="all", types=list(simulator.types))
    return DataGenerator(
        simulator=simulator,
        observation=observation,
        design=instantiate(cfg.design, _convert_="all"),
        summary=instantiate(cfg.summary, _convert_="all"),
        seed=seed,
        workers=int(cfg.workers),
    )


def build_components(cfg: DictConfig) -> Components:
    """Instantiate the inference model, its data generator and the sbi method."""
    generator = _generator(cfg, cfg, int(cfg.seed))
    method = instantiate(cfg.inference.method, _convert_="all", device=cfg.device, seed=cfg.seed)
    return Components(generator.simulator, generator, method)


def build_data_generator(cfg: DictConfig, model: DataGenerator) -> DataGenerator:
    """Instantiate the data-generating model and check it is comparable with ``model``.

    Data and inference model must share the time unit and the measured labels; the
    number of *true* types, transitions, founders and parameters may differ. The data
    generator draws from :func:`data_seed`, the inference generator from ``seed``.
    """
    data = _generator(cfg, cfg.data, data_seed(cfg))
    if data.simulator.time_unit != model.simulator.time_unit:
        raise ValueError(
            f"Data model counts time in {data.simulator.time_unit}, inference model in "
            f"{model.simulator.time_unit}: horizons would mean different things"
        )
    if tuple(data.observation.labels) != tuple(model.observation.labels):
        raise ValueError(
            f"Data labels {data.observation.labels} differ from inference labels "
            f"{model.observation.labels}; map true types to labels with data.observation"
        )
    return data


def is_well_specified(cfg: DictConfig) -> bool:
    """Return True when the data model config equals the inference model config."""
    keys = ("simulator", "law", "observation")
    return all(OmegaConf.to_container(cfg.data[k], resolve=True) ==
               OmegaConf.to_container(cfg[k], resolve=True) for k in keys)  # fmt: skip


def reference_parameters(cfg: DictConfig, simulator: BranchingSimulator) -> np.ndarray:
    """Physical parameters of the reference dataset under the data model ``simulator``.

    ``truth.source=reference`` merges ``data.law.reference``, ``data.simulator.reference``
    (entries for types the law lacks are ignored) and ``truth.values``, later winning;
    fixed values of the data law fill in automatically. ``truth.source=prior`` draws
    one sample from the data model's prior, seeded by :func:`data_seed`.
    """
    if cfg.truth.source == "prior":
        seed_torch(seed_for(data_seed(cfg), "truth"))
        return simulator.decode(simulator.prior().sample((1,)).numpy()[0])
    if cfg.truth.source != "reference":
        raise ValueError("truth.source must be 'reference' or 'prior'")

    def section(node) -> dict:
        return flatten(OmegaConf.to_container(node, resolve=True)) if node is not None else {}

    sim_ref = section(cfg.data.simulator.get("reference"))
    values = {
        **section(cfg.data.law.get("reference")),
        **{k: v for k, v in sim_ref.items() if k in simulator.parameter_names},
        **section(cfg.truth.get("values")),
    }
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
    data_gen = build_data_generator(cfg, gen)
    prior = sim.prior()
    well_specified = is_well_specified(cfg)

    # 1. Reference ("observed") dataset from the data model.
    truth = reference_parameters(cfg, data_gen.simulator)
    detection: list[dict] = []
    reference = data_gen.dataset(truth, "reference", stats=detection)
    if reference is None:
        raise RuntimeError("Reference parameters exceed simulation limits; adjust truth/limits")
    x_obs = gen.summarize(reference)
    np.savez_compressed(
        out / "reference.npz",
        truth=truth,
        truth_names=np.array(data_gen.simulator.parameter_names),
        labels=np.array(gen.observation.labels),
        x=x_obs,
        **{f"counts_h{c.horizon:g}": c.counts for c in reference},
    )
    LOG.info("Reference clones: %s", {c.horizon: c.counts.sum(0).tolist() for c in reference})
    LOG.info("Reference detection: %s", detection)
    if not well_specified:
        LOG.info("Misspecified: data %s/%s, inference %s/%s", cfg.data.simulator.name,
                 cfg.data.law.name, cfg.simulator.name, cfg.law.name)  # fmt: skip

    # 2. Train on the inference model.
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

    # 3. Recovery at the reference truth (parameters matched by name).
    t_sample = time.perf_counter()
    samples_theta = fitted.sample(
        x_obs, int(cfg.budget.posterior_samples), seed_for(cfg.seed, "ref")
    )
    sample_seconds = time.perf_counter() - t_sample
    samples = sim.decode(samples_theta)
    seed_torch(seed_for(cfg.seed, "prior-reference"))
    prior_samples = sim.decode(prior.sample((4096,)).numpy())
    truth_by_name = dict(zip(data_gen.simulator.parameter_names, truth, strict=True))
    matched = np.array([truth_by_name.get(n, np.nan) for n in sim.parameter_names])
    free = set(sim.space.free_names)
    rows = [
        {**row, "inferred": row["parameter"] in free}
        for row in recovery_table(samples, matched, prior_samples, sim.parameter_names)
    ]
    pd.DataFrame(rows).to_csv(out / "recovery.csv", index=False)
    pd.DataFrame(samples, columns=sim.parameter_names).to_csv(
        out / "posterior_samples.csv.gz", index=False
    )
    shown = [i for i, n in enumerate(sim.parameter_names) if n in free]
    plots.posterior_marginals(
        samples[:, shown], prior_samples[:, shown], matched[shown],
        [sim.parameter_names[i] for i in shown], out / "figures" / "posterior.png",
    )  # fmt: skip
    scored = [r for r in rows if r["inferred"] and np.isfinite(r["truth"])]
    interior = [r for r in scored if not r["at_boundary"]]  # boundary truths: no coverage

    metrics: dict = {
        "simulator": cfg.simulator.name,
        "law": cfg.law.name,
        "data_simulator": cfg.data.simulator.name,
        "data_law": cfg.data.law.name,
        "well_specified": well_specified,
        "seed": int(cfg.seed),
        "data_seed": data_seed(cfg),
        "inference": cfg.inference.name,
        "rounds": comp.method.rounds,
        "simulations": int(cfg.budget.simulations),
        "design": {
            "name": gen.design.name,
            "horizons": list(gen.design.horizons),
            "clones": list(gen.design.clones),
        },
        "reference_detection": detection,
        "n_free_parameters": int(sim.theta_dim),
        "x_dim": int(fitted.feature_mask.sum()),
        "training": fitted.history,
        "sampling_seconds": sample_seconds,
        "recovery": {
            "scored_parameters": len(scored),
            "mean_abs_error": float(np.mean([r["abs_error"] for r in scored]))
            if scored
            else float("nan"),
            "mean_contraction": float(
                np.nanmean([r["contraction"] for r in rows if r["inferred"]])
            ),
            "coverage90": float(np.mean([r["covered"] for r in interior]))
            if interior
            else float("nan"),
            "per_parameter": rows,
        },
    }

    # 4. Posterior predictive check: does the fitted inference model reproduce x_obs?
    n_ppc = int(cfg.budget.get("ppc_samples", 0))
    if n_ppc > 0:
        ppc, x_pred = posterior_predictive_check(gen, samples_theta[:n_ppc], x_obs, fitted)
        metrics["ppc"] = ppc
        np.savez_compressed(out / "ppc.npz", x_pred=x_pred, x_obs=x_obs)

    # 5. Calibration of the inference model (amortised posteriors only).
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
        "Done in %.1fs: MAE=%.4f contraction=%.3f PPC outside95=%s -> %s",
        metrics["total_seconds"], metrics["recovery"]["mean_abs_error"],
        metrics["recovery"]["mean_contraction"],
        metrics.get("ppc", {}).get("fraction_outside_95", "n/a"), out,
    )  # fmt: skip
    return metrics


def load_run(run_dir: str | Path) -> tuple[DictConfig, Components, FittedPosterior]:
    """Reload a finished run: its config, inference components and trained posterior."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return cfg, build_components(cfg), FittedPosterior.load(run_dir / "posterior.pt")


def infer_measured(run_dir: str | Path, data: str | Path, out: str | Path, n: int = 4096) -> Path:
    """Condition a trained posterior on measured counts and save posterior samples.

    Args:
        run_dir: Directory of a finished amortised run.
        data: CSV with one row per clone: a ``horizon`` column plus one count column
            per measured label (e.g. ``A``, ``B``).
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
    labels = list(gen.observation.labels)
    table = pd.read_csv(data)
    missing = {"horizon", *labels} - set(table.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {sorted(missing)}")
    extra = set(table["horizon"]) - set(gen.design.horizons)
    if extra:
        raise ValueError(f"Horizons {sorted(extra)} were not part of the trained design")
    cohorts = []
    for horizon, size in gen.design.cohorts:
        rows = table[np.isclose(table["horizon"], horizon)]
        if len(rows) == 0:
            raise ValueError(f"No clones at horizon {horizon:g}; trained design: {gen.design}")
        if len(rows) != size:
            LOG.warning("Horizon %g: %d clones measured, %d trained", horizon, len(rows), size)
        cohorts.append(Cohort(horizon, rows[labels].to_numpy(dtype=np.int64)))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    samples = sim.decode(fitted.sample(gen.summarize(cohorts), n, seed_for(cfg.seed, "measured")))
    frame = pd.DataFrame(samples, columns=sim.parameter_names)
    frame.to_csv(out / "posterior_samples.csv.gz", index=False)
    frame.describe(percentiles=[0.05, 0.5, 0.95]).T.to_csv(out / "posterior_summary.csv")
    free = [i for i, name in enumerate(sim.parameter_names) if name in set(sim.space.free_names)]
    seed_torch(seed_for(cfg.seed, "prior-reference"))
    prior_samples = sim.decode(sim.prior().sample((4096,)).numpy())
    plots.posterior_marginals(
        samples[:, free], prior_samples[:, free], None,
        [sim.parameter_names[i] for i in free], out / "posterior.png",
    )  # fmt: skip
    return out
