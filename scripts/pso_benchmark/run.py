"""Benchmark: multistart L-BFGS-B vs PSO alone vs PSO + refinement (exact GW point estimates).

Small simulated reference datasets (generated as ``bio-sbi`` does), several optimiser
seeds per case, 2 worker threads. Writes ``results.json`` and ``results.md`` next to
this file. Run from the repository root with a thread cap::

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
        uv run python scripts/pso_benchmark/run.py [--seeds 5] [--quick] [--cases NAME ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_gw_optimise import dataset  # noqa: E402

from bio_sbi.inference.gw_optimise import optimise  # noqa: E402
from bio_sbi.pipeline import build_components  # noqa: E402

HERE = Path(__file__).resolve().parent

CASES = {
    # name: (Hydra overrides, objective)
    "rev_h46_c128_d0": (["law=reversible_ab", "design.horizons=[4,6]", "design.clones=128",
                         "data_seed=0"], "mle"),
    "rev_h46_c128_d1": (["law=reversible_ab", "design.horizons=[4,6]", "design.clones=128",
                         "data_seed=1"], "mle"),
    "two_h46_c128_d0": (["law=two_type", "design.horizons=[4,6]", "design.clones=128",
                         "data_seed=0"], "mle"),
    "two_h35_c64_d1": (["law=two_type", "design.horizons=[3,5]", "design.clones=64",
                        "data_seed=1"], "mle"),
    "two_h48_c256_d2": (["law=two_type", "design.horizons=[4,8]", "design.clones=256",
                         "data_seed=2"], "mle"),
    # physical MAP with an informative Dirichlet (alpha > 1) on both rows
    "two_h46_c128_d0_map": (["law=two_type", "design.horizons=[4,6]", "design.clones=128",
                             "data_seed=0", "+law.concentration.A.death=2",
                             "+law.concentration.B.to_A=3", "+law.concentration.B.death=2"],
                            "map"),
}  # fmt: skip

METHODS = {
    "multistart": {"method": "multistart", "starts": 8},
    "pso": {"method": "pso", "refine_top": 0},
    "pso+refine": {"method": "pso", "refine_top": 4},
}
COMMON = {"restarts": 2, "swarm": 24, "max_iter": 100, "refine_maxiter": 500, "workers": 2,
          "log_every": 0, "validate": False}  # fmt: skip


def eval_cost(lik, n=20):
    """Seconds per objective-only and per gradient evaluation (single thread)."""
    rng = np.random.default_rng(0)
    X = rng.beta(lik.prior_a, lik.prior_b, size=(n, lik.dim))
    out = {}
    for grad in (False, True):
        t = time.perf_counter()
        for x in X:
            lik.log_likelihood(x, grad=grad)
        out["gradient" if grad else "objective_only"] = (time.perf_counter() - t) / n
    return out


def main() -> None:
    """Run every case x method x seed; write results.json / results.md."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="2 cases, 2 seeds")
    parser.add_argument(
        "--cases", nargs="*", default=None, help="Rerun only these cases (merged into results.json)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    cases = dict(list(CASES.items())[:2]) if args.quick else CASES
    seeds = range(2 if args.quick else args.seeds)
    results = {}
    if args.cases:
        cases = {k: CASES[k] for k in args.cases}
        results = json.loads((HERE / "results.json").read_text())
    for name, (overrides, objective) in cases.items():
        ns = argparse.Namespace(run_dir=None, overrides=["inference.method.threads=2", *overrides])
        cfg, cohorts, truth = dataset(ns)
        comp = build_components(cfg)
        sim = comp.simulator
        lik = comp.method.likelihood(sim, comp.generator.observation, cohorts)
        free = list(sim.space.free_names)
        truth_theta = sim.encode(np.array([truth[n] for n in sim.parameter_names])).astype(float)
        case = {"overrides": overrides, "objective": objective, "dim": lik.dim,
                "grids": [(g.Lx, g.Ly) for g in lik.grids],
                "clones": [h.clones for h in lik.data], "eval_seconds": eval_cost(lik),
                "truth": {n: truth[n] for n in free},
                "truth_log_likelihood": float(lik.log_likelihood(truth_theta, False)[0]),
                "runs": {}}  # fmt: skip
        for method, opts in METHODS.items():
            runs = []
            for seed in seeds:
                t = time.perf_counter()
                r = optimise(lik, seed=seed, objective=objective, **COMMON, **opts)
                runs.append({
                    "seed": seed, "value": r["value"], "seconds": time.perf_counter() - t,
                    "evaluations": r["evaluations"],
                    "log_likelihood": r["best"]["log_likelihood"],
                    "log_prior": r["best"]["log_prior"],
                    "physical": {n: r["best"]["physical"][n] for n in free},
                    "distinct_values": [c["value"] for c in r["candidates"] if c["distinct"]],
                    "stop_reasons": [p["stop_reason"] for p in r.get("pso", [])],
                    "near_zero": r["best"]["boundary"]["probabilities_near_zero"],
                })  # fmt: skip
                print(f"{name} {method} seed {seed}: {r['value']:.4f} "
                      f"{runs[-1]['seconds']:.1f}s {r['evaluations']}", flush=True)  # fmt: skip
            case["runs"][method] = runs
        # numerical validation of the best PSO + refinement solution (seed 0)
        best = optimise(lik, seed=0, objective=objective, **{**COMMON, "validate": True},
                        **METHODS["pso+refine"])  # fmt: skip
        v = best["validation"]
        case["validation"] = {
            "flags": v["flags"], "best_runner_up_gap": v["best_runner_up_gap"],
            "max_numerical_change": v["max_numerical_change"],
            "grid_changes": [g.get("max_abs_change") for g in v["grids"]],
            "floor_changes": [f["max_abs_change"] for f in v["floors"]],
            "floored_cells": v["cells"]["floored_cells"],
            "grid_check_max_rel_error": max(g["max_rel_error"] for g in v["grid_check"]),
            "competitive": [{"value": c["value"], **{n: c["physical"][n] for n in free}}
                            for c in best["candidates"] if c["distinct"]],
        }  # fmt: skip
        results[name] = case
        (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_markdown(results)


def write_markdown(results: dict) -> None:
    """Summary tables (results.md)."""
    lines = ["# PSO benchmark (generated by run.py)", ""]
    lines.append("| case | dim | method | best | median gap | max gap | hit | mean s | "
                 "obj-only evals | grad evals |")  # fmt: skip
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, case in results.items():
        best = max(r["value"] for runs in case["runs"].values() for r in runs)
        for method, runs in case["runs"].items():
            gaps = np.array([best - r["value"] for r in runs])
            lines.append(
                f"| {name} | {case['dim']} | {method} | {max(r['value'] for r in runs):.3f} | "
                f"{np.median(gaps):.2g} | {gaps.max():.2g} | {np.sum(gaps < 1e-2)}/{len(runs)} | "
                f"{np.mean([r['seconds'] for r in runs]):.1f} | "
                f"{np.mean([r['evaluations']['objective_only'] for r in runs]):.0f} | "
                f"{np.mean([r['evaluations']['gradient'] for r in runs]):.0f} |"
            )
    lines += ["", "gap = best value over all runs of the case minus the run's value (nats); "
              "hit = runs within 0.01 nats of that best.", ""]  # fmt: skip
    lines.append("| case | s/eval obj-only | s/eval grad | truth logL | flags | grid change | "
                 "floor change | floored cells |")  # fmt: skip
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, case in results.items():
        v = case["validation"]
        lines.append(
            f"| {name} | {case['eval_seconds']['objective_only']:.4f} | "
            f"{case['eval_seconds']['gradient']:.4f} | {case['truth_log_likelihood']:.3f} | "
            f"{', '.join(v['flags']) or '-'} | {max(c or 0 for c in v['grid_changes']):.1e} | "
            f"{max(v['floor_changes']):.1e} | {v['floored_cells']} |"
        )
    lines += ["", "## Best estimates vs truth (pso+refine, seed 0)", ""]
    for name, case in results.items():
        r = case["runs"]["pso+refine"][0]
        lines.append(f"**{name}** (value {r['value']:.3f}, near-zero: "
                     f"{', '.join(r['near_zero']) or '-'})")  # fmt: skip
        lines.append("")
        lines.append("| parameter | truth | estimate | spread over seeds/methods |")
        lines.append("|---|---|---|---|")
        allp = [x["physical"] for runs in case["runs"].values() for x in runs]
        for n, t in case["truth"].items():
            vals = [p[n] for p in allp]
            lines.append(f"| {n} | {t:.3f} | {r['physical'][n]:.3f} | "
                         f"{min(vals):.3f} - {max(vals):.3f} |")  # fmt: skip
        comp = case["validation"]["competitive"]
        if len(comp) > 1:
            lines.append("")
            lines.append("Distinct refined candidates (value; B row): " + "; ".join(
                f"{c['value']:.3f} ({', '.join(f'{k}={c[k]:.3f}' for k in c if k.startswith('B.'))})"
                for c in comp))  # fmt: skip
        lines.append("")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
