r"""Benchmark of exact GW point estimators: multistart L-BFGS-B, PSO, SGD family.

Small simulated reference datasets (generated as ``bio-sbi`` does), several optimiser
seeds per case, 2 worker threads, every method rerun in the same session. Writes
``results.json`` and ``results.md`` (and ``trace_*.png``) next to this file. Run from
the repository root with a thread cap::

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \\
        uv run python scripts/pso_benchmark/run.py [--seeds 5] [--quick] [--cases NAME ...] \\
        [--skip-main] [--skip-lr] [--skip-trace] [--skip-minibatch]

Parts: ``main`` (every case x method x seed), ``lr`` (learning-rate sensitivity of
SGD without polish on two cases), ``trace`` (best objective vs likelihood calls and vs
wall time, one worker, seed 0, two cases), ``minibatch`` (cost of one gradient step vs
minibatch size and the minibatch gradient noise on a larger design).

Methods ``sgd-*`` are run once with the L-BFGS-B polish; the unpolished variant
(``adam``, ``momentum``, ...) is read from the same run (best SGD value, evaluations
and seconds before the polish), so both share the SGD trajectory exactly.
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

from bio_sbi.inference.gw_exact import law_and_jac  # noqa: E402
from bio_sbi.inference.gw_optimise import PointObjective, minibatch_weights, optimise  # noqa: E402
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

SGD = {"method": "sgd", "starts": 8, "refine_top": 4}  # same 8 prior starts as multistart
METHODS = {
    "multistart": {"method": "multistart", "starts": 8},
    "pso": {"method": "pso", "refine_top": 0},
    "pso+refine": {"method": "pso", "refine_top": 4},
    # SGD family: Adam / momentum, sticks (box) or logits, full batch or 25% minibatches
    "adam+polish": {**SGD},
    "adam-logit+polish": {**SGD, "sgd_coords": "logit", "sgd_lr": 0.1},
    "momentum+polish": {**SGD, "sgd_optimizer": "momentum", "sgd_lr": 0.01},
    "adam-mb25+polish": {**SGD, "sgd_batch": 0.25},
}
COMMON = {"restarts": 2, "swarm": 24, "max_iter": 100, "refine_maxiter": 500, "workers": 2,
          "log_every": 0, "validate": False}  # fmt: skip
#: Learning-rate sensitivity (no polish, 500 steps unless stated)
LR_CASES = ["rev_h46_c128_d0", "two_h46_c128_d0"]
LR_CONFIGS = {
    **{f"adam lr={lr:g}": {"sgd_lr": lr} for lr in (0.005, 0.02, 0.05, 0.1)},
    **{f"momentum lr={lr:g}": {"sgd_optimizer": "momentum", "sgd_lr": lr}
       for lr in (0.003, 0.01, 0.03, 0.1)},
    "adam lr=0.05 x2000 steps": {"sgd_lr": 0.05, "sgd_iter": 2000},
    "adam-logit lr=0.1": {"sgd_coords": "logit", "sgd_lr": 0.1},
    "adam-logit lr=0.3": {"sgd_coords": "logit", "sgd_lr": 0.3},
}  # fmt: skip
TRACE_CASES = ["rev_h46_c128_d0", "two_h46_c128_d0"]
TRACE_METHODS = ["multistart", "pso+refine", "adam+polish", "momentum+polish",
                 "adam-mb25+polish", "adam-logit+polish"]  # fmt: skip
MINIBATCH_DESIGNS = {
    "two_h4-6-8-10_c128": ["law=two_type", "design.horizons=[4,6,8,10]", "design.clones=128",
                           "data_seed=0"],
    "two_h4-6-8-10_c1024": ["law=two_type", "design.horizons=[4,6,8,10]", "design.clones=1024",
                            "data_seed=0"],
}  # fmt: skip
MINIBATCH_FRACTIONS = [0.01, 0.05, 0.25, 1.0]


def load(overrides):
    """``(lik, simulator, truth by name)`` of a simulated reference dataset."""
    ns = argparse.Namespace(run_dir=None, overrides=["inference.method.threads=2", *overrides])
    cfg, cohorts, truth = dataset(ns)
    comp = build_components(cfg)
    lik = comp.method.likelihood(comp.simulator, comp.generator.observation, cohorts)
    return lik, comp.simulator, truth


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


def record(lik, r: dict, seed: int, seconds: float, free) -> dict:
    """Summary of one optimise() result."""
    return {
        "seed": seed, "value": r["value"], "seconds": seconds, "evaluations": r["evaluations"],
        "log_likelihood": r["best"]["log_likelihood"], "log_prior": r["best"]["log_prior"],
        "physical": {n: r["best"]["physical"][n] for n in free},
        "distinct_values": [c["value"] for c in r["candidates"] if c["distinct"]],
        "stop_reasons": [p["stop_reason"] for p in r.get("pso", []) or r.get("sgd", [])],
        "near_zero": r["best"]["boundary"]["probabilities_near_zero"],
    }  # fmt: skip


def unpolished(lik, r: dict, seed: int, free) -> dict:
    """Summarise the SGD part of a method=sgd result (best SGD endpoint, cost before the polish)."""
    i = int(np.argmax([s["best_value"] for s in r["sgd"]]))
    theta = np.array(r["sgd"][i]["best_theta"])
    phys = lik.space.decode(theta)
    names = list(lik.space.names)
    obj = PointObjective(lik, r["settings"]["objective"], r["settings"]["boundary_eps"])
    gap = np.minimum(theta - obj.lower, obj.upper - theta)
    return {
        "seed": seed, "value": r["sgd_best_value"], "seconds": r["sgd_seconds"],
        "evaluations": r["evaluations_before_polish"],
        "physical": {n: float(phys[names.index(n)]) for n in free},
        "stop_reasons": [s["stop_reason"] for s in r["sgd"]],
        "near_zero": [n for n in free if phys[names.index(n)] <= 1e-6],
        "min_probability": float(min(phys[names.index(n)] for n in free)),
        "min_stick_gap": float(gap.min()) if len(gap) else None,
        "steps": [s["iterations"] for s in r["sgd"]],
    }  # fmt: skip


# ---------------------------------------------------------------------- parts
def run_main(results, cases, seeds):
    """Every case x method x seed (+ validation of the best multistart solution)."""
    for name, (overrides, objective) in cases.items():
        lik, sim, truth = load(overrides)
        free = list(sim.space.free_names)
        truth_theta = sim.encode(np.array([truth[n] for n in sim.parameter_names])).astype(float)
        case = {"overrides": overrides, "objective": objective, "dim": lik.dim,
                "grids": [(g.Lx, g.Ly) for g in lik.grids],
                "clones": [h.clones for h in lik.data], "eval_seconds": eval_cost(lik),
                "truth": {n: truth[n] for n in free},
                "truth_log_likelihood": float(lik.log_likelihood(truth_theta, False)[0]),
                "runs": {}}  # fmt: skip
        for method, opts in METHODS.items():
            runs, raw = [], []
            for seed in seeds:
                t = time.perf_counter()
                r = optimise(lik, seed=seed, objective=objective, **COMMON, **opts)
                runs.append(record(lik, r, seed, time.perf_counter() - t, free))
                if opts["method"] == "sgd":
                    raw.append(unpolished(lik, r, seed, free))
                print(f"{name} {method} seed {seed}: {r['value']:.4f} "
                      f"{runs[-1]['seconds']:.1f}s {r['evaluations']}"
                      + (f" | sgd {raw[-1]['value']:.4f}" if raw else ""), flush=True)  # fmt: skip
            if raw:
                case["runs"][method.removesuffix("+polish")] = raw
            case["runs"][method] = runs
        # numerical validation of the best multistart solution (seed 0)
        best = optimise(lik, seed=0, objective=objective, **{**COMMON, "validate": True},
                        **METHODS["multistart"])  # fmt: skip
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
        results["cases"][name] = case
        save(results)


def run_lr(results, seeds):
    """SGD without polish at several learning rates / optimisers / coordinates."""
    out = results.setdefault("lr_sensitivity", {})
    for name in LR_CASES:
        overrides, objective = CASES[name]
        lik, sim, _ = load(overrides)
        out[name] = {}
        for label, conf in LR_CONFIGS.items():
            vals, secs, steps = [], [], []
            for seed in seeds:
                t = time.perf_counter()
                r = optimise(lik, seed=seed, objective=objective, **COMMON,
                             **{**SGD, "sgd_polish": False, **conf})  # fmt: skip
                vals.append(r["sgd_best_value"])
                secs.append(time.perf_counter() - t)
                steps.append(r["evaluations"]["gradient"])
            out[name][label] = {"values": vals, "seconds": secs, "grad_evals": steps}
            print(f"lr {name} {label}: {np.round(vals, 3)} {np.mean(secs):.1f}s", flush=True)
        save(results)


def traced(lik):
    """Record (time, value or nan, grad) of every likelihood call; returns the list."""
    rec = []
    orig = type(lik).log_likelihood

    def wrapper(theta, grad=True, weights=None):
        out = orig(lik, theta, grad=grad, weights=weights)
        rec.append((time.perf_counter(), out[0] if weights is None else np.nan, bool(grad)))
        return out

    lik.log_likelihood = wrapper
    return rec


def run_trace(results):
    """Best value vs likelihood calls and wall time (seed 0, one worker)."""
    out = results.setdefault("traces", {})
    for name in TRACE_CASES:
        overrides, objective = CASES[name]
        lik, sim, _ = load(overrides)
        out[name] = {}
        for method in TRACE_METHODS:
            rec = traced(lik)
            t0 = time.perf_counter()
            optimise(lik, seed=0, objective=objective, **{**COMMON, "workers": 1},
                     **METHODS[method])  # fmt: skip
            del lik.log_likelihood
            t = np.array([r[0] - t0 for r in rec])
            v = np.array([r[1] for r in rec])
            best = np.fmax.accumulate(np.where(np.isnan(v), -np.inf, v))
            grad = np.array([r[2] for r in rec])
            # keep the improvements only (plus the last call)
            keep = np.r_[True, best[1:] > best[:-1]]
            keep[-1] = True
            out[name][method] = {
                "seconds": t[keep].tolist(), "calls": (np.flatnonzero(keep) + 1).tolist(),
                "grad_calls": np.cumsum(grad)[keep].tolist(), "best": best[keep].tolist(),
                "total_calls": len(rec), "total_seconds": float(t[-1]),
            }  # fmt: skip
            print(f"trace {name} {method}: {best[-1]:.4f} {len(rec)} calls {t[-1]:.1f}s", flush=True)
        save(results)


def timed(f, n: int) -> float:
    """Median wall time of ``n`` calls of ``f``."""
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def run_minibatch(results, reps=15):
    """Cost of a gradient step vs minibatch fraction, and minibatch gradient noise."""
    out = results.setdefault("minibatch", {})
    for name, overrides in MINIBATCH_DESIGNS.items():
        out[name] = minibatch_design(overrides, reps)
        print(f"minibatch {name}: {json.dumps(out[name]['fractions'])}", flush=True)
        save(results)


def minibatch_design(overrides, reps: int) -> dict:
    """Minibatch cost / noise on one design (see :func:`run_minibatch`)."""
    lik, sim, truth = load(overrides)
    obj = PointObjective(lik)
    x = sim.encode(np.array([truth[n] for n in sim.parameter_names])).astype(float)
    rng = np.random.default_rng(0)
    P, dP, rows = lik.jacobian(x)
    law = sum(timed(lambda h=h: law_and_jac(lik.model, P, dP, rows, h.generations, h.grid),
                    reps) for h in lik.data)  # fmt: skip
    # noise at the truth (near the optimum: small full gradient) and half-way to the
    # centre of the stick box (far from it)
    points = {"truth": x, "midway": 0.5 * x + 0.25}
    g_full = {k: obj.value_and_grad(p)[1] for k, p in points.items()}
    case = {"overrides": overrides, "clones": [h.clones for h in lik.data],
            "cells": [len(h.weight) for h in lik.data],
            "grids": [(h.grid.Lx, h.grid.Ly) for h in lik.data],
            "objective_only_s": timed(lambda: obj.value(x), reps),
            "gradient_full_s": timed(lambda: obj.value_and_grad(x), reps),
            "law_and_jac_s": law, "fractions": {}}  # fmt: skip
    cells = [np.repeat(np.arange(len(h.weight)), h.weight.astype(np.int64)) for h in lik.data]
    for frac in MINIBATCH_FRACTIONS:
        w = minibatch_weights(lik, rng, frac, cells)
        case["fractions"][str(frac)] = {
            "sample_s": timed(lambda f=frac: minibatch_weights(lik, rng, f, cells), reps),
            "gradient_s": timed(lambda w=w: obj.value_and_grad(x, weights=w), reps),
        }
        for k, p in points.items():
            G = np.array([obj.value_and_grad(p, weights=minibatch_weights(lik, rng, frac,
                                                                          cells))[1]
                          for _ in range(40)])  # fmt: skip
            g = g_full[k]
            cos = G @ g / (np.linalg.norm(G, axis=1) * np.linalg.norm(g))
            case["fractions"][str(frac)][k] = {
                "rel_grad_error": float(np.median(np.linalg.norm(G - g, axis=1))
                                        / np.linalg.norm(g)),
                "cosine_to_full": float(np.median(cos)),
            }  # fmt: skip
    return case


def save(results):
    """Write results.json."""
    (HERE / "results.json").write_text(json.dumps(results, indent=1, default=float))


# ---------------------------------------------------------------------- report
ORDER = ["multistart", "pso", "pso+refine", "adam", "adam+polish", "momentum",
         "momentum+polish", "adam-logit", "adam-logit+polish", "adam-mb25", "adam-mb25+polish"]  # fmt: skip


def write_markdown(results: dict) -> None:
    """Summary tables (results.md)."""
    cases = results.get("cases", {})
    L = ["# Exact GW point-estimate benchmark (generated by run.py)", "",
         "Every method on the same cases, seeds 0-4, 2 worker threads, same session. "
         "`gap` = best value over all runs of all methods for the case minus the run's value "
         "(nats); `hit` = gap < 0.01. Methods without `+polish` are the SGD part of the "
         "`+polish` run (value, seconds, evaluations before the L-BFGS-B polish). SGD: 8 "
         "starts = the multistart prior draws, 500 steps, cosine schedule to 1% of the rate, "
         "stop if the best improved < 1e-3 nats in 100 steps; `adam` lr 0.02 (sticks), "
         "`momentum` lr 0.01, `adam-logit` lr 0.1 in logit(stick), `adam-mb25` = Adam on "
         "25% minibatches of clones per horizon. Polish: L-BFGS-B on the 4 best distinct "
         "endpoints.", ""]  # fmt: skip
    L.append("## Summary over all cases")
    L.append("")
    L.append("| method | hits | worst gap | median s | mean obj-only evals | mean grad evals |")
    L.append("|---|---|---|---|---|---|")
    for method in ORDER:
        gaps, secs, ev_o, ev_g = [], [], [], []
        for case in cases.values():
            if method not in case["runs"]:
                continue
            best = max(r["value"] for runs in case["runs"].values() for r in runs)
            for r in case["runs"][method]:
                gaps.append(best - r["value"])
                secs.append(r["seconds"])
                ev_o.append(r["evaluations"]["objective_only"])
                ev_g.append(r["evaluations"]["gradient"])
        if gaps:
            g = np.array(gaps)
            L.append(f"| {method} | {np.sum(g < 1e-2)}/{len(g)} | {g.max():.3g} | "
                     f"{np.median(secs):.1f} | {np.mean(ev_o):.0f} | {np.mean(ev_g):.0f} |")  # fmt: skip
    L += ["", "## Per case", ""]
    L.append("| case | dim | method | hit | worst gap | mean gap (sd) | best value | mean s | "
             "obj-only evals | grad evals |")  # fmt: skip
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, case in cases.items():
        best = max(r["value"] for runs in case["runs"].values() for r in runs)
        for method in ORDER:
            runs = case["runs"].get(method)
            if not runs:
                continue
            gaps = np.array([best - r["value"] for r in runs])
            L.append(
                f"| {name} | {case['dim']} | {method} | {np.sum(gaps < 1e-2)}/{len(runs)} | "
                f"{gaps.max():.3g} | {gaps.mean():.3g} ({gaps.std():.2g}) | "
                f"{max(r['value'] for r in runs):.3f} | "
                f"{np.mean([r['seconds'] for r in runs]):.1f} | "
                f"{np.mean([r['evaluations']['objective_only'] for r in runs]):.0f} | "
                f"{np.mean([r['evaluations']['gradient'] for r in runs]):.0f} |"
            )
    L += ["", "## Boundary optima and logit coordinates", "",
          "Best SGD endpoint before polish: free probabilities <= 1e-6 (count, mean over "
          "seeds), smallest free probability and smallest stick distance to the box "
          "(median over seeds); polished best: probabilities <= 1e-6.", "",
          "| case | method | p<=1e-6 (SGD) | min p (SGD) | min stick gap (SGD) | "
          "p<=1e-6 (polished) | value lost vs polish |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, case in cases.items():
        for m in ("adam", "adam-logit"):
            if m not in case["runs"]:
                continue
            raw, pol = case["runs"][m], case["runs"][m + "+polish"]
            L.append(f"| {name} | {m} | {np.mean([len(r['near_zero']) for r in raw]):.1f} | "
                     f"{np.median([r['min_probability'] for r in raw]):.1e} | "
                     f"{np.median([r['min_stick_gap'] for r in raw]):.1e} | "
                     f"{np.mean([len(r['near_zero']) for r in pol]):.1f} | "
                     f"{np.median([p['value'] - r['value'] for p, r in zip(pol, raw, strict=True)]):.3g} |")  # fmt: skip
    if cases:
        L += ["", "| case | s/eval obj-only | s/eval grad | truth logL | flags | grid change | "
              "floor change | floored cells |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, case in cases.items():
            v = case["validation"]
            L.append(
                f"| {name} | {case['eval_seconds']['objective_only']:.4f} | "
                f"{case['eval_seconds']['gradient']:.4f} | {case['truth_log_likelihood']:.3f} | "
                f"{', '.join(v['flags']) or '-'} | {max(c or 0 for c in v['grid_changes']):.1e} | "
                f"{max(v['floor_changes']):.1e} | {v['floored_cells']} |"
            )
    lr = results.get("lr_sensitivity", {})
    if lr:
        L += ["", "## Learning-rate sensitivity (SGD alone, no polish, 8 starts, seeds 0-4)", "",
              "gap to the case's best value from the main table (nats).", "",
              "| case | config | hits | median gap | worst gap | mean s | grad evals |",
              "|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, confs in lr.items():
            best = max(r["value"] for runs in cases[name]["runs"].values() for r in runs) \
                if name in cases else max(max(c["values"]) for c in confs.values())
            for label, c in confs.items():
                g = best - np.array(c["values"])
                L.append(f"| {name} | {label} | {np.sum(g < 1e-2)}/{len(g)} | "
                         f"{np.median(g):.3g} | {g.max():.3g} | {np.mean(c['seconds']):.1f} | "
                         f"{np.mean(c['grad_evals']):.0f} |")  # fmt: skip
    tr = results.get("traces", {})
    if tr:
        L += ["", "## Convergence traces (seed 0, one worker)", "",
              "Gap of the best objective found so far to the case's best value after the "
              "given wall time (s) / likelihood calls (objective-only + gradient). "
              "Minibatch gradient calls count as calls but carry no full value. "
              "Plots: `trace_<case>.png`.", ""]  # fmt: skip
        checkpoints_t = [0.1, 0.3, 1, 3, 10]
        checkpoints_n = [50, 200, 1000, 4000]
        for name, methods in tr.items():
            best = max(r["value"] for runs in cases[name]["runs"].values() for r in runs) \
                if name in cases else max(m["best"][-1] for m in methods.values())
            L.append(f"**{name}**")
            L.append("")
            L.append("| method | " + " | ".join(f"{c} s" for c in checkpoints_t) + " | "
                     + " | ".join(f"{c} calls" for c in checkpoints_n) + " | total |")  # fmt: skip
            L.append("|---" * (2 + len(checkpoints_t) + len(checkpoints_n)) + "|")
            for method, m in methods.items():
                def at(xs, c, m=m, best=best):
                    i = np.searchsorted(np.asarray(xs), c, side="right") - 1
                    return "-" if i < 0 else f"{best - m['best'][i]:.3g}"
                L.append(f"| {method} | " + " | ".join(at(m["seconds"], c) for c in checkpoints_t)
                         + " | " + " | ".join(at(m["calls"], c) for c in checkpoints_n)
                         + f" | {best - m['best'][-1]:.3g} ({m['total_calls']} calls, "
                         f"{m['total_seconds']:.1f} s) |")  # fmt: skip
            L.append("")
    mb = results.get("minibatch", {})
    if mb:
        L += ["", "## Minibatch cost and noise (two_type, generations 4/6/8/10, truth)", "",
              "Median seconds of one call (Numba 2 threads). `law+jac` = the pgf recursion "
              "+ FFTs alone (all horizons); minibatch steps differ from a full step only "
              "in the cell weights of the final dot product.", "",
              "Gradient noise (40 minibatches): median |g_b - g| / |g| and median cosine "
              "between g_b and the full g, at the truth (near the MLE, small |g|) and "
              "half-way from the truth to the centre of the stick box.", "",
              "| design | grids | cells | obj-only s | full grad s | law+jac s | fraction | "
              "sample s | minibatch grad s | rel. error / cosine (truth) | "
              "rel. error / cosine (midway) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for name, c in mb.items():
            for frac, f in c["fractions"].items():
                L.append(f"| {name} | {c['grids']} | {sum(c['cells'])} | "
                         f"{c['objective_only_s']:.4f} | {c['gradient_full_s']:.4f} | "
                         f"{c['law_and_jac_s']:.4f} | {frac} | {f['sample_s']:.5f} | "
                         f"{f['gradient_s']:.4f} | "
                         f"{f['truth']['rel_grad_error']:.3g} / {f['truth']['cosine_to_full']:.2f} | "
                         f"{f['midway']['rel_grad_error']:.3g} / {f['midway']['cosine_to_full']:.2f} |")  # fmt: skip
    (HERE / "results.md").write_text("\n".join(L) + "\n")


def plot_traces(results: dict) -> None:
    """Write trace_<case>.png (gap to the best value vs calls and vs wall time)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = results.get("cases", {})
    for name, methods in results.get("traces", {}).items():
        best = max(r["value"] for runs in cases[name]["runs"].values() for r in runs) \
            if name in cases else max(m["best"][-1] for m in methods.values())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for method, m in methods.items():
            gap = np.maximum(best - np.array(m["best"]), 1e-4)
            for ax, xs in zip(axes, (m["calls"], m["seconds"]), strict=True):
                ax.step(np.r_[xs, m["total_calls"] if ax is axes[0] else m["total_seconds"]],
                        np.r_[gap, gap[-1]], where="post", label=method)  # fmt: skip
        for ax, lab in zip(axes, ("likelihood calls", "wall time (s, 1 worker)"), strict=True):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(lab)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("gap to best value (nats, floored at 1e-4)")
        axes[1].legend(fontsize=8)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(HERE / f"trace_{name}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    """Run the selected parts; write results.json / results.md / trace plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="2 cases, 2 seeds")
    parser.add_argument("--cases", nargs="*", default=None, help="Main part: only these cases")
    for part in ("main", "lr", "trace", "minibatch"):
        parser.add_argument(f"--skip-{part}", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rewrite results.md / plots")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    path = HERE / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    if "cases" not in results:  # old format (PSO benchmark only): start over
        results = {"cases": {}}
    if not args.report_only:
        seeds = range(2 if args.quick else args.seeds)
        cases = dict(list(CASES.items())[:2]) if args.quick else CASES
        if args.cases:
            cases = {k: CASES[k] for k in args.cases}
        if not args.skip_main:
            run_main(results, cases, seeds)
        if not args.skip_lr:
            run_lr(results, seeds)
        if not args.skip_trace:
            run_trace(results)
        if not args.skip_minibatch:
            run_minibatch(results)
    write_markdown(results)
    plot_traces(results)


if __name__ == "__main__":
    main()
