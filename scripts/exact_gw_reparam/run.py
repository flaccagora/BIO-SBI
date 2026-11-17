"""Benchmark exact_gw sampler coordinates / SMC start on XG2 cells (one run per call).

Usage (from the repo root; 4 threads)::

    uv run python scripts/exact_gw_reparam/run.py --cell T1_n1024_s0 --arm brow+probit \
        --out /home/matteo/BIO-SBI/results/exact_gw/reparam_bench

Arms (all: adjoint kernel, the XG2 preset's init_starts / warmup, ESS / R-hat stopping
rule on with its defaults, 4 chains, 4 threads):

* ``baseline`` - default coordinates (u = logit theta);
* ``probit`` / ``brow`` / ``brow+probit`` - ``inference.method.reparam``;
* ``smc`` / ``smc+<spec>`` - tempered SMC from the prior, NUTS chains started from the
  ``beta = 1`` population (``smc.mode=start``) in the given coordinates;
* ``smcfinal`` / ``smcfinal+<spec>`` - the SMC chains are the draws (``smc.mode=final``).

``--seed`` changes the sampler seed (``seed`` of the method; the dataset stays the XG2
one). Writes ``<out>/<cell>/<arm>[_seed<k>]/`` with ``draws.npz`` (theta draws per chain,
physical draws, SMC population), ``diagnostics.json`` and ``run.log``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))
from harness import load  # noqa: E402


def arm_settings(arm: str) -> tuple[str, dict | None]:
    """``(reparam spec, smc options or None)`` of an arm name."""
    head, _, spec = arm.partition("+")
    if head == "baseline":
        return "none", None
    if head in ("smc", "smcfinal"):
        return spec or "none", {"enabled": True, "mode": "start" if head == "smc" else "final"}
    return arm, None


def main() -> None:
    """Run one cell x arm."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cell", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None, help="sampler seed (default: XG2's)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smc", nargs="*", default=[], help="extra smc options key=value")
    a = p.parse_args()
    name = a.arm.replace("+", "_") + (f"_seed{a.seed}" if a.seed is not None else "")
    out = a.out / a.cell / name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.FileHandler(out / "run.log", mode="w"),
                                  logging.StreamHandler()])  # fmt: skip
    method, lik, cfg, truth = load(a.cell)
    spec, smc = arm_settings(a.arm)
    method.reparam = spec
    method.threads = a.threads
    method.log_every = 100
    if a.seed is not None:
        method.seed = a.seed
    stop = {**method.stop, "enabled": True}
    if smc is not None:
        for kv in a.smc:
            k, v = kv.split("=", 1)
            smc[k] = type(method.smc[k])(v) if method.smc[k] is not None else v
        smc = {**method.smc, **smc}
    t0 = time.perf_counter()
    theta, diag = method.run(lik, stop=stop, smc=smc)
    wall = time.perf_counter() - t0
    chains, n = int(diag["chains"]), int(diag["draws_per_chain"])
    theta = theta.reshape(chains, n, -1)
    extra = {}
    if method.last_smc is not None:
        coords = method.coordinates(lik)
        extra["smc_population"] = coords.theta(method.last_smc.draws)
        extra["smc_population_log_post"] = method.last_smc.log_post.reshape(-1)
    np.savez_compressed(out / "draws.npz", theta=theta, physical=lik.space.decode(theta),
                        names=np.array(lik.space.names), truth=truth, **extra)  # fmt: skip
    diag["wall_seconds"] = wall
    diag["likelihood_eval_seconds"] = lik.eval_seconds
    diag["likelihood_evals_total"] = lik.evals
    diag["likelihood_grad_evals_total"] = lik.grad_evals
    diag["arm"], diag["cell"], diag["reparam_spec"] = a.arm, a.cell, spec
    diag["smc_options"] = smc
    diag["sampler_seed"] = method.seed
    diag["config_method"] = OmegaConf.to_container(cfg.inference.method, resolve=True)
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1, default=float))
    logging.info("done %s %s in %.0fs", a.cell, a.arm, wall)


if __name__ == "__main__":
    main()
