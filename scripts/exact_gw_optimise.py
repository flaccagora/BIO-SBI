"""Point estimate (MLE / physical MAP) of the exact GW likelihood, without NUTS.

Writes ``OUT/optimisation.json`` (see ``bio_sbi.inference.gw_optimise``); settings are
``inference.method.optimise.*`` of ``inference=exact_gw`` (Hydra overrides). The dataset
is either a simulated reference dataset, generated exactly as ``bio-sbi`` does for the
same config, or the ``reference.npz`` of a finished run (``--run-dir``)::

    uv run python scripts/exact_gw_optimise.py --out opt/mle law=reversible_ab \
        design.horizons=[4,6] design.clones=128 inference.method.threads=2
    uv run python scripts/exact_gw_optimise.py --out opt/map law=two_type \
        inference.method.optimise.objective=map '+law.concentration={A: {death: 2}}'
    uv run python scripts/exact_gw_optimise.py --run-dir RUN --out RUN/opt \
        inference.method.optimise.method=multistart

Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf, open_dict

from bio_sbi.pipeline import build_components, build_data_generator, reference_parameters
from bio_sbi.summaries import Cohort

LOG = logging.getLogger("exact_gw_optimise")


def dataset(args) -> tuple:
    """``(cfg, cohorts, truth by name or None)``."""
    if args.run_dir:
        run = Path(args.run_dir)
        cfg = OmegaConf.load(run / ".hydra" / "config.yaml")
        if cfg.inference.name != "exact_gw":
            with initialize_config_module("bio_sbi.conf", version_base="1.3"):
                node = compose("config", overrides=["inference=exact_gw"]).inference
            with open_dict(cfg):
                cfg.inference = node
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
        z = np.load(run / "reference.npz")
        cohorts = [Cohort(float(k[len("counts_h") :]), z[k]) for k in z.files
                   if k.startswith("counts_h")]  # fmt: skip
        cohorts.sort(key=lambda c: c.horizon)
        truth = dict(zip(map(str, z["truth_names"]), map(float, z["truth"]), strict=True))
        return cfg, cohorts, truth
    with initialize_config_module("bio_sbi.conf", version_base="1.3"):
        cfg = compose("config", overrides=["inference=exact_gw", *args.overrides])
    gen = build_components(cfg).generator
    data_gen = build_data_generator(cfg, gen)
    for attempt in range(int(cfg.truth.get("max_draws", 50))):
        truth = reference_parameters(cfg, data_gen.simulator, attempt)
        cohorts = data_gen.dataset(truth, "reference", stats=[])
        if cohorts is not None:
            break
    names = data_gen.simulator.parameter_names
    return cfg, cohorts, dict(zip(names, map(float, truth), strict=True))


def main(argv=None) -> None:
    """Parse arguments, optimise, write ``optimisation.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--run-dir", default=None, help="Finished run (reference.npz)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    cfg, cohorts, truth = dataset(args)
    comp = build_components(cfg)
    method, sim = comp.method, comp.simulator
    lik = method.likelihood(sim, comp.generator.observation, cohorts)
    result = method.point_estimate(lik)
    free = set(sim.space.free_names)
    best = result["best"]["physical"]
    if truth:
        result["truth"] = {n: truth.get(n) for n in sim.parameter_names}
        if all(n in truth for n in free):
            # only meaningful where the parameters are identified (see the README)
            result["truth_log_likelihood"] = float(
                lik.log_likelihood(sim.encode(np.array([truth[n] for n in sim.parameter_names]))
                                   .astype(float), grad=False)[0]
            )  # fmt: skip
            result["abs_error"] = {n: abs(best[n] - truth[n]) for n in sim.space.free_names}
    result["config"] = {"law": cfg.law.name, "seed": int(cfg.seed),
                        "horizons": list(cfg.design.horizons), "clones": cfg.design.clones,
                        "grids": [(g.Lx, g.Ly) for g in lik.grids]}  # fmt: skip
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "optimisation.json").write_text(json.dumps(result, indent=2, default=float))
    LOG.info("%s = %.4f (%s): %s", result["objective"], result["value"], out / "optimisation.json",
             {n: round(best[n], 4) for n in sim.space.free_names})  # fmt: skip
    if result.get("validation", {}).get("flags"):
        LOG.warning("validation flags: %s", result["validation"]["flags"])


if __name__ == "__main__":
    main()
