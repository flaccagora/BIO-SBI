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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from . import plots
from .design import Design
from .evaluation import (
    calibration,
    calibration_metrics,
    posterior_predictive_check,
    ppc_metrics,
    predictive_table,
    recovery_table,
)
from .inference import FittedPosterior, InferenceMethod
from .provenance import write_manifest
from .rng import seed_for, seed_torch
from .simulate import DataGenerator
from .simulators import BranchingSimulator
from .simulators.law import flatten
from .summaries import Cohort

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

    Data and inference model must share the *effective* time unit (e.g. Bellman-Harris
    hours and Galton-Watson with ``hours_per_generation`` are both "hours") and the
    measured labels; the
    number of *true* types, transitions, founders and parameters may differ. The data
    generator draws from :func:`data_seed`, the inference generator from ``seed``.
    """
    data = _generator(cfg, cfg.data, data_seed(cfg))
    if data.simulator.time_unit != model.simulator.time_unit:
        raise ValueError(
            f"Data model counts time in {data.simulator.time_unit}, inference model in "
            f"{model.simulator.time_unit}: horizons would mean different things (a "
            "Galton-Watson model can count hours via simulator=galton_watson_hours)"
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


def reference_parameters(
    cfg: DictConfig, simulator: BranchingSimulator, attempt: int = 0
) -> np.ndarray:
    """Physical parameters of the reference dataset under the data model ``simulator``.

    ``truth.source=reference`` merges ``data.law.reference``, ``data.simulator.reference``
    (entries for types the law lacks are ignored) and ``truth.values``, later winning;
    fixed values of the data law fill in automatically. ``truth.source=prior`` draws
    one sample from the data model's prior, seeded by :func:`data_seed`; ``attempt > 0``
    gives the next independent draw (used when a draw cannot produce a reference
    dataset, see :func:`run_experiment`).
    """
    if cfg.truth.source == "prior":
        keys = ("truth",) if attempt == 0 else ("truth", attempt)  # attempt 0: as before
        seed_torch(seed_for(data_seed(cfg), *keys))
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

    # 1. Reference ("observed") dataset from the data model. A prior-drawn truth can be
    # unusable (clones never detected, or above the simulation limits): redraw it,
    # reproducibly from data_seed, up to truth.max_draws times and record the count.
    max_draws = int(cfg.truth.get("max_draws", 50)) if cfg.truth.source == "prior" else 1
    for attempt in range(max_draws):
        truth = reference_parameters(cfg, data_gen.simulator, attempt)
        detection: list[dict] = []
        reference = data_gen.dataset(truth, "reference", stats=detection)
        if reference is not None:
            break
    if reference is None:
        raise RuntimeError(
            f"No usable reference dataset after {max_draws} truth draw(s): the parameters "
            "exceed the simulation limits or no clone is detectable; adjust truth/limits"
        )
    truth_draws = attempt + 1
    # Expected living cells per TRUE type (incl. types the markers cannot see), per horizon.
    expected_counts = {}
    for horizon in data_gen.design.horizons:
        try:
            counts = data_gen.simulator.expected_counts(truth, horizon)
            expected_counts[f"{horizon:g}"] = dict(
                zip(data_gen.simulator.types, map(float, counts), strict=True)
            )
        except NotImplementedError:
            break
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

    # Exact-likelihood methods (inference=exact_gw / exact_bh) condition on the raw
    # reference counts (MCMC / importance sampling) instead of training on simulations.
    exact = bool(getattr(comp.method, "exact", False))
    if exact:
        fitted = comp.method.fit_data(sim, gen.observation, reference, gen.x_dim)
    else:
        # `prior` (CPU) is used for NumPy-side draws; the trainer gets a prior on its device.
        fitted = comp.method.fit(
            sim.prior(cfg.device), simulate, int(cfg.budget.simulations), x_obs,
            log_dir=out / "tensorboard",
        )  # fmt: skip
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
    # Share of the (NPE) posterior inside the prior support at x_obs; ~0 under severe
    # misspecification, when sampling may return few or no samples (see FittedPosterior).
    support_mass = fitted.support_mass(x_obs, seed=seed_for(cfg.seed, "support"))
    n_drawn = len(samples_theta)
    if n_drawn == 0:
        LOG.warning("No posterior samples inside the prior support: recovery and PPC are NaN")
        samples_theta = np.full((2, sim.theta_dim), np.nan)
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
    if n_drawn > 0:
        plots.posterior_marginals(
            samples[:, shown], prior_samples[:, shown], matched[shown],
            [sim.parameter_names[i] for i in shown], out / "figures" / "posterior.png",
        )  # fmt: skip
    scored = [r for r in rows if r["inferred"] and np.isfinite(r["truth"])]
    interior = [r for r in scored if not r["at_boundary"]]
    scalar_names = {s.name for s in sim.space.scalars}
    prob_scored = [r for r in scored if r["parameter"] not in scalar_names]
    scalar_scored = [
        r for r in scored if r["parameter"] in scalar_names and r["truth"]
    ]  # boundary truths: no coverage

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
        "simulations": 0 if exact else int(cfg.budget.simulations),
        "design": {
            "name": gen.design.name,
            "horizons": list(gen.design.horizons),
            "clones": list(gen.design.clones),
        },
        "reference_detection": detection,
        "truth_draws": truth_draws,
        "reference_expected_counts": expected_counts or None,
        "n_free_parameters": int(sim.theta_dim),
        "x_dim": int(fitted.feature_mask.sum()),
        "training": fitted.history,
        "sampling_seconds": sample_seconds,
        "posterior_samples_drawn": int(n_drawn),
        "posterior_support_mass": support_mass,
        "recovery": {
            "scored_parameters": len(scored),
            # Probabilities and scalars (e.g. lifetimes in hours) have different units:
            # `mean_abs_error` averages probabilities only; scalars get a relative error.
            "mean_abs_error": float(np.mean([r["abs_error"] for r in prob_scored]))
            if prob_scored
            else float("nan"),
            "mean_rel_error_scalars": float(
                np.mean([r["abs_error"] / abs(r["truth"]) for r in scalar_scored])
            )
            if scalar_scored
            else None,
            "mean_abs_error_all": float(np.mean([r["abs_error"] for r in scored]))
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
    if exact:
        metrics[comp.method.name] = fitted.diagnostics

    # 4. Posterior predictive check: does the fitted inference model reproduce x_obs?
    n_ppc = int(cfg.budget.get("ppc_samples", 0))
    if n_ppc > 0 and n_drawn > 0:
        ppc, x_pred = posterior_predictive_check(gen, samples_theta[:n_ppc], x_obs, fitted)
        metrics["ppc"] = ppc
        np.savez_compressed(out / "ppc.npz", x_pred=x_pred, x_obs=x_obs)

    # 5. Calibration of the inference model (amortised posteriors only).
    cases = int(cfg.budget.sbc_cases)
    if exact and cfg.inference.get("sbc_cases") is not None:
        cases = int(cfg.inference.sbc_cases)  # exact methods: one posterior run per case
    if cases > 0 and exact:
        seed_torch(seed_for(cfg.seed, "sbc-theta"))
        theta_sbc = prior.sample((cases,)).numpy()
        x_sbc, data_sbc = gen.simulate_theta(theta_sbc, "sbc")
        ranks, info = comp.method.calibrate(
            sim, gen.observation, theta_sbc, data_sbc, int(cfg.budget.sbc_samples)
        )
        cal = calibration_metrics(
            ranks, sim.theta_names, info["draws_per_case"], invalid=info["invalid_cases"],
            excluded=info["excluded_grid"],
        )  # fmt: skip
        metrics["calibration"] = {**cal, comp.method.name: info}
        np.savez_compressed(out / "sbc.npz", ranks=ranks, theta=theta_sbc, x=x_sbc)
        if len(ranks):
            plots.sbc_ranks(
                ranks, info["draws_per_case"], sim.theta_names, out / "figures" / "sbc.png"
            )
    elif cases > 0 and fitted.amortized:
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


def measured_cohorts(table: pd.DataFrame, design: Design, labels: list[str]) -> list[Cohort]:
    """Split a measured ``horizon,<labels>`` table into cohorts in design order.

    Horizons absent from the trained design, or design horizons without clones, raise
    ``ValueError``; a clone count different from the design only logs a warning.
    """
    missing = {"horizon", *labels} - set(table.columns)
    if missing:
        raise ValueError(f"CSV is missing columns {sorted(missing)}")
    counts = table[labels].to_numpy()
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts % 1):
        raise ValueError("Counts must be finite non-negative integers")
    extra = {float(h) for h in table["horizon"] if not np.any(np.isclose(h, design.horizons))}
    if extra:
        raise ValueError(f"Horizons {sorted(extra)} were not part of the trained design")
    cohorts = []
    for horizon, size in design.cohorts:
        rows = table[np.isclose(table["horizon"], horizon)]
        if len(rows) == 0:
            raise ValueError(f"No clones at horizon {horizon:g}; trained design: {design}")
        if len(rows) != size:
            LOG.warning("Horizon %g: %d clones measured, %d trained", horizon, len(rows), size)
        cohorts.append(Cohort(horizon, rows[labels].to_numpy(dtype=np.int64)))
    return cohorts


def infer_measured(
    run_dir: str | Path, data: str | Path, out: str | Path, n: int = 4096, ppc: int = 200
) -> Path:
    """Condition a trained posterior on measured counts and check the fit.

    Writes to ``out``: ``posterior_samples.csv.gz``, ``posterior_summary.csv``,
    ``posterior.png``, ``metrics.json`` (posterior support mass, per-parameter
    posterior mean/sd/quantiles/contraction, PPC metrics), ``ppc.csv`` (per summary
    feature: observed value, predictive mean/sd/95% interval, tail probability,
    z-score) and ``ppc.npz``. The PPC simulates the *measured* number of clones per
    horizon, which may differ from the trained design.

    Args:
        run_dir: Directory of a finished amortised run.
        data: CSV with one row per clone: a ``horizon`` column plus one count column
            per measured label (e.g. ``A``, ``B``).
        out: Output directory.
        n: Number of posterior samples.
        ppc: Posterior predictive datasets (0 disables the check).

    Returns:
        The output directory.
    """
    cfg, comp, fitted = load_run(run_dir)
    if not fitted.amortized:
        raise ValueError("Measured inference needs an amortised (rounds=1) posterior")
    sim, gen = comp.simulator, comp.generator
    labels = list(gen.observation.labels)
    cohorts = measured_cohorts(pd.read_csv(data), gen.design, labels)
    x_obs = gen.summarize(cohorts)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    support_mass = fitted.support_mass(x_obs, seed=seed_for(cfg.seed, "measured-support"))
    if support_mass is not None and support_mass < 0.5:
        LOG.warning("Only %.1f%% of the posterior lies inside the prior support: the model "
                    "may not explain these data", 100 * support_mass)  # fmt: skip
    samples_theta = fitted.sample(x_obs, n, seed_for(cfg.seed, "measured"))
    n_drawn = len(samples_theta)
    samples = sim.decode(samples_theta) if n_drawn else np.empty((0, len(sim.parameter_names)))
    frame = pd.DataFrame(samples, columns=sim.parameter_names)
    frame.to_csv(out / "posterior_samples.csv.gz", index=False)
    frame.describe(percentiles=[0.05, 0.5, 0.95]).T.to_csv(out / "posterior_summary.csv")
    free_names = set(sim.space.free_names)
    free = [i for i, name in enumerate(sim.parameter_names) if name in free_names]
    seed_torch(seed_for(cfg.seed, "prior-reference"))
    prior_samples = sim.decode(sim.prior().sample((4096,)).numpy())
    if n_drawn:
        plots.posterior_marginals(
            samples[:, free], prior_samples[:, free], None,
            [sim.parameter_names[i] for i in free], out / "posterior.png",
        )  # fmt: skip
    else:
        LOG.warning("No posterior samples inside the prior support: no summary or PPC")

    levels = (0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)
    parameters = {}
    for i, name in enumerate(sim.parameter_names):
        prior_var = float(prior_samples[:, i].var())
        row = {"inferred": name in free_names, "prior_mean": float(prior_samples[:, i].mean()),
               "prior_sd": float(np.sqrt(prior_var))}  # fmt: skip
        if n_drawn:
            values = samples[:, i]
            row.update(
                mean=float(values.mean()),
                sd=float(values.std()),
                quantiles={
                    f"{q:g}": float(v)
                    for q, v in zip(levels, np.quantile(values, levels), strict=True)
                },
                contraction=float(1 - values.var() / prior_var) if prior_var > 0 else None,
            )
        parameters[name] = row

    measured_design = Design(
        [c.horizon for c in cohorts], [len(c.counts) for c in cohorts], name="measured"
    )
    metrics: dict = {
        "kind": "measured",
        "run_dir": str(Path(run_dir).resolve()),
        "data": str(Path(data).resolve()),
        "simulator": cfg.simulator.name,
        "law": cfg.law.name,
        "inference": cfg.inference.name,
        "trained_design": {
            "horizons": list(gen.design.horizons),
            "clones": list(gen.design.clones),
        },
        "measured": [
            {
                "horizon": c.horizon,
                "clones": len(c.counts),
                **{f"mean_{k}": float(c.counts[:, j].mean()) for j, k in enumerate(labels)},
                "mean_total": float(c.counts.sum(1).mean()),
                "max_total": int(c.counts.sum(1).max()),
                "empty_clones": int((c.counts.sum(1) == 0).sum()),
            }
            for c in cohorts
        ],
        "posterior_samples_requested": int(n),
        "posterior_samples_drawn": int(n_drawn),
        "posterior_support_mass": support_mass,
        "parameters": parameters,
    }

    # Posterior predictive check at the measured clone counts.
    n_ppc = min(int(ppc), n_drawn)
    if n_ppc > 0:
        ppc_gen = replace(gen, design=measured_design)
        x_pred, _ = ppc_gen.simulate_theta(samples_theta[:n_ppc], "measured-ppc")
        valid = int(np.all(np.isfinite(x_pred), axis=1).sum())
        if valid == 0:
            LOG.warning("All %d predictive datasets hit the simulation limits: no PPC", n_ppc)
            metrics["ppc"] = {"samples": 0, "requested": n_ppc}
        else:
            names = gen.summary.feature_names(measured_design.horizons, labels)
            table = predictive_table(x_pred, x_obs, fitted.feature_mask, names)
            metrics["ppc"] = {**ppc_metrics(table, valid), "requested": n_ppc}
            metrics["ppc_table"] = table
            pd.DataFrame(table).to_csv(out / "ppc.csv", index=False)
        np.savez_compressed(out / "ppc.npz", x_pred=x_pred, x_obs=x_obs)
    _write_json(out / "metrics.json", metrics)
    LOG.info(
        "Measured fit: %d samples, support mass %s, PPC outside95=%s -> %s",
        n_drawn, support_mass, metrics.get("ppc", {}).get("fraction_outside_95", "n/a"), out,
    )  # fmt: skip
    return out
