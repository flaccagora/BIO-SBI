"""Maximum-likelihood estimates for finished exact_gw runs, next to their NUTS posterior.

For every run folder (``metrics.json`` + ``.hydra/``) under the given roots:

* rebuilds the exact likelihood of the run's reference dataset (regenerated from the
  run's config and checked bit for bit against ``reference.npz``);
* maximises the log likelihood (no prior) by multistart L-BFGS-B in stick coordinates
  (bounds [1e-9, 1 - 1e-9]), from data-driven and prior starts: the NUTS per-chain
  means, the posterior mean, and ``--prior-starts`` prior draws. The truth is NOT a start
  (fair estimator); a separate optimisation started at the truth flags a missed optimum;
* writes ``mle.json`` (MLE per parameter, log likelihood at MLE / truth / posterior mean,
  distinct optima, per-start results, flags) and redraws ``figures/posterior.png`` with
  the MLE as a dashed line (the original figure is kept once as
  ``figures/posterior_nuts.png``).

Usage:
  uv run python scripts/exact_gw_mle.py results/exact_gw/data_size [--workers 16]
      [--prior-starts 8] [--force]
Each worker uses one thread (set OMP/NUMBA/MKL threads to 1, done here). A run with an
up-to-date ``mle.json`` is skipped unless ``--force``.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

LO, HI = 1e-9, 1 - 1e-9


def _setup(run: Path):
    import pandas as pd
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from bio_sbi.pipeline import build_components, build_data_generator, reference_parameters

    overrides = list(OmegaConf.load(run / ".hydra" / "overrides.yaml"))
    with initialize_config_module("bio_sbi.conf", version_base=None):
        cfg = compose("config", overrides=overrides)
    comp = build_components(cfg)
    data = build_data_generator(cfg, comp.generator)
    truth = reference_parameters(cfg, data.simulator)
    saved = np.load(run / "reference.npz")
    if not np.allclose(saved["truth"], truth):
        # prior-drawn truths may need redraws (truth.max_draws): use the saved truth
        truth = saved["truth"]
    ref = data.dataset(truth, "reference")
    if ref is None or not all(
        np.array_equal(c.counts, saved[f"counts_h{c.horizon:g}"]) for c in ref
    ):
        raise RuntimeError(f"{run}: regenerated reference differs from reference.npz")
    sim = comp.simulator
    lik = comp.method.likelihood(sim, comp.generator.observation, ref)
    names = list(sim.parameter_names)
    post = pd.read_csv(run / "posterior_samples.csv.gz")[names].to_numpy()
    return cfg, sim, lik, names, truth, post


def fit_run(run: str, prior_starts: int) -> dict:
    import torch
    from scipy.optimize import minimize

    from bio_sbi import plots
    from bio_sbi.rng import seed_for, seed_torch

    t0 = time.perf_counter()
    run = Path(run)
    cfg, sim, lik, names, truth, post = _setup(run)
    metrics = json.loads((run / "metrics.json").read_text())
    free = list(sim.space.free_names)
    fidx = [names.index(n) for n in free]

    def enc(physical):
        return np.clip(sim.encode(physical), LO, HI)

    def ll(theta):
        return float(lik.log_likelihood(np.clip(theta, LO, HI), grad=False)[0])

    def neg(theta):
        v, g = lik.log_likelihood(theta)
        return -v, -g

    def optimise(start):
        r = minimize(neg, start, jac=True, method="L-BFGS-B", bounds=[(LO, HI)] * len(start),
                     options={"maxiter": 1000})  # fmt: skip
        return {"ll": float(-r.fun), "theta": r.x, "success": bool(r.success),
                "iterations": int(r.nit), "message": str(r.message)}  # fmt: skip

    starts = {"posterior_mean": enc(post.mean(0))}
    chain_means = (metrics.get("exact_gw") or {}).get("chain_means") or {}
    for c in range(len(next(iter(chain_means.values()), []))):
        phys = post.mean(0).copy()
        for n, v in chain_means.items():
            phys[names.index(n)] = v[c]
        # chain means are stored rounded (4 decimals): renormalise each outcome row
        for t in sim.law.types:
            idx = [i for i, n in enumerate(names) if n.startswith(f"{t}.")]
            phys[idx] = np.clip(phys[idx], 1e-9, None) / np.clip(phys[idx], 1e-9, None).sum()
        starts[f"chain_{c}"] = enc(phys)
    seed_torch(seed_for(int(cfg.seed), "mle-starts"))
    for k, th in enumerate(sim.prior().sample((prior_starts,)).numpy()):
        starts[f"prior_{k}"] = np.clip(th, 1e-6, 1 - 1e-6)
    results = {name: optimise(st) for name, st in starts.items()}
    best_name = max(results, key=lambda k: results[k]["ll"])
    best = results[best_name]
    from_truth = optimise(enc(truth))
    mle = sim.decode(np.clip(best["theta"], LO, HI))
    distinct = sorted({round(r["ll"], 2) for r in results.values()}, reverse=True)
    out = {
        "run": str(run),
        "parameters": free,
        "mle": {n: float(mle[i]) for n, i in zip(free, fidx, strict=True)},
        "truth": {n: float(truth[i]) for n, i in zip(free, fidx, strict=True)},
        "posterior_mean": {n: float(post[:, i].mean()) for n, i in zip(free, fidx, strict=True)},
        "ll_mle": best["ll"],
        "ll_truth": ll(enc(truth)),
        "ll_posterior_mean": ll(enc(post.mean(0))),
        "best_start": best_name,
        "distinct_optima_ll": distinct[:8],
        "starts_within_0.01_of_best": int(sum(r["ll"] > best["ll"] - 0.01 for r in results.values())),
        "truth_start_ll": from_truth["ll"],
        "missed_optimum": bool(from_truth["ll"] > best["ll"] + 0.01),
        "at_bound": [n for n, i in zip(free, fidx, strict=True) if mle[i] < 1e-6],
        "starts": {k: {kk: vv for kk, vv in r.items() if kk != "theta"} for k, r in results.items()},
        "likelihood_evals": lik.evals,
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (run / "mle.json").write_text(json.dumps(out, indent=2))

    # Redraw posterior.png with the MLE (same prior draws as the pipeline).
    fig_dir = run / "figures"
    if (fig_dir / "posterior.png").exists() and not (fig_dir / "posterior_nuts.png").exists():
        shutil.copy2(fig_dir / "posterior.png", fig_dir / "posterior_nuts.png")
    seed_torch(seed_for(int(cfg.seed), "prior-reference"))
    prior_samples = sim.decode(sim.prior().sample((4096,)).numpy())
    plots.posterior_marginals(
        post[:, fidx], prior_samples[:, fidx], truth[fidx], free, fig_dir / "posterior.png",
        markers={"MLE": mle[fidx]},
    )  # fmt: skip
    return {"run": str(run), "seconds": out["seconds"], "missed": out["missed_optimum"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prior-starts", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    runs = []
    for root in args.roots:
        for m in sorted(root.rglob("metrics.json")):
            run = m.parent
            if not (run / ".hydra" / "overrides.yaml").exists():
                continue
            done = run / "mle.json"
            if done.exists() and not args.force and done.stat().st_mtime > m.stat().st_mtime:
                continue
            runs.append(str(run))
    print(f"{len(runs)} runs to fit with {args.workers} workers", flush=True)
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fit_run, r, args.prior_starts): r for r in runs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
                flag = "  MISSED OPTIMUM (truth start better)" if res["missed"] else ""
                print(f"[{k}/{len(runs)}] {res['seconds']:.0f}s {res['run']}{flag}", flush=True)
            except Exception as exc:  # report and continue
                failed += 1
                print(f"[{k}/{len(runs)}] FAILED {futs[f]}: {exc!r}", flush=True)
    print(f"done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
