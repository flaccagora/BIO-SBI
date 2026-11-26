"""Structured truths D0 / S0 / M: maximum likelihood and NUTS at n = 512 per endpoint.

MLE (``mle``): D0 datasets 0-9 of every truth, models ``free`` (12 probabilities) and
``nodeath`` (death fixed at 0); the multistart of ``dfcommon.fit_mle`` (as the
misspecification study, whose base fits of the original truths on the paired datasets are
the reference). NUTS (``nuts``): dataset 0, flat priors, as the misspecification study's
runs (adjoint kernel, 4 chains from L-BFGS optima of 8 prior draws, 300 warmup, tree depth
<= 7, stop at R-hat 1.01 / bulk ESS 200 / tail ESS 100 or 500 draws per chain), for
``D0:free``, ``D0:nodeath`` and ``orig:free`` (paired baseline).

    uv run python scripts/death_free/run.py mle --jobs all --workers 16
    uv run python scripts/death_free/run.py nuts --jobs D0:free,D0:nodeath,orig:free \
        --workers 4 --threads 4

Outputs ``results/exact_gw/death_free/{mle,nuts}/<kind>_<model>/<gt>_s<seed>.json``
(restartable: finished or locked jobs are skipped).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dfcommon as D  # noqa: E402

C = D.C
SUMMARIES = ("rho", "stableA", "m_BA", "B.div")


def _lock(out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out.with_suffix(".lock"), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def truth_dict(gt: str, kind: str) -> dict:
    P = D.truth(gt, kind)
    return {**C.pdict(P), **C.summaries(P)}


# ---------------------------------------------------------------------- MLE
def mle_job(gt: str, seed: int, kind: str, model: str) -> tuple:
    t0 = time.perf_counter()
    out = D.OUT / "mle" / f"{kind}_{model}" / f"{gt}_s{seed}.json"
    if not _lock(out):
        return gt, seed, f"{kind}:{model}", 0.0, "locked"
    try:
        data = D.dataset(gt, seed, kind)
        lik = C.make_lik(data)
        f = D.fit_mle(lik, np.random.default_rng([seed, 99]), model)
        P = D.truth(gt, kind)
        res = {"gt": gt, "seed": seed, "model": model, "kind": kind,
               "truth": truth_dict(gt, kind), "P": C.pdict(f["P"]), "summ": C.summaries(f["P"]),
               "ll": f["ll"], "ll_truth": D.loglik_at(lik, P),
               "start": f["start"], "n_tied": f["n_tied"], "second_ll": f["second_ll"],
               "mean_cells": [float(c.counts.sum(1).mean()) for c in data],
               "seconds": time.perf_counter() - t0}  # fmt: skip
        out.write_text(json.dumps(res, default=float, indent=1))
        return gt, seed, f"{kind}:{model}", res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".err").write_text(traceback.format_exc())
        return gt, seed, f"{kind}:{model}", 0.0, repr(exc)


# ---------------------------------------------------------------------- NUTS
def make_sim(model: str):
    import yaml

    from bio_sbi.simulators import GaltonWatson

    law = yaml.safe_load((D.ROOT / "src" / "bio_sbi" / "conf" / "law" / "two_type.yaml").read_text())
    if model == "nodeath":
        law["fixed"] = {"founder": {"A": 1.0}, "A": {"death": 0.0}, "B": {"death": 0.0}}
    elif model == "noswitch":
        law["fixed"] = {"founder": {"A": 1.0}, "A": {"to_B": 0.0}, "B": {"to_A": 0.0}}
    elif model != "free":
        raise ValueError(f"NUTS: model {model} not expressible as a law (rows cannot be tied)")
    sim = GaltonWatson(law)
    assert list(sim.space.names)[:12] == D.NAMES, sim.space.names
    return sim


def nuts_job(gt: str, kind: str, model: str, threads: int, seed: int = 0) -> tuple:
    from bio_sbi.inference.exact_gw import ExactGW, ExactLikelihood

    t0 = time.perf_counter()
    out = D.OUT / "nuts" / f"{kind}_{model}" / f"{gt}_s{seed}.json"
    if not _lock(out):
        return gt, f"{kind}:{model}", 0.0, "locked"
    try:
        data = D.dataset(gt, seed, kind)
        sim = make_sim(model)
        lik = ExactLikelihood(sim, D.dl.observation(None), data, kernel="adjoint")
        m = ExactGW(kernel="adjoint", threads=threads, chains=4, init_starts=8, warmup=300,
                    draws=500, log_every=0, seed=seed, max_tree_depth=7,
                    stop={"enabled": True, "ess": 200.0, "tail_ess": 100.0,
                          "max_draws": 500})  # fmt: skip
        theta, diag = m.run(lik, key=("death_free", kind, model, gt, seed))
        phys = sim.space.decode(np.asarray(theta))[:, :12]
        Ps = phys.reshape(-1, 2, 6)
        summ = [C.summaries(P) for P in Ps]
        cols = {n: phys[:, i] for i, n in enumerate(D.NAMES)}
        for k in SUMMARIES:
            cols[k] = np.array([s[k] for s in summ])
        truth = truth_dict(gt, kind)
        res = {"gt": gt, "kind": kind, "model": model, "seed": seed, "stats": {}}
        for k, v in cols.items():
            mu, sd = float(v.mean()), float(v.std())
            q05, q50, q95 = (float(x) for x in np.quantile(v, [0.05, 0.5, 0.95]))
            t = truth[k]
            res["stats"][k] = {"mean": mu, "sd": sd, "q05": q05, "median": q50, "q95": q95,
                               "truth": t, "z": (mu - t) / sd if sd > 0 else None,
                               "cover90": bool(q05 - 1e-12 <= t <= q95 + 1e-12)}  # fmt: skip
        res["draws"] = int(len(phys))
        res["convergence"] = {k: diag.get(k) for k in (
            "max_rank_rhat", "min_ess_bulk", "min_ess_tail", "max_mean_spread", "divergences",
            "max_tree_depth_hits", "draws_per_chain", "step_size")}  # fmt: skip
        st = diag.get("stop") or {}
        res["stop"] = {k: st.get(k) for k in ("reason", "failed", "draws_per_chain")}
        res["seconds"] = time.perf_counter() - t0
        np.savez_compressed(out.with_suffix(".npz"), phys=phys.astype(np.float32))
        out.write_text(json.dumps(res, default=float, indent=1))
        return gt, f"{kind}:{model}", res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".err").write_text(traceback.format_exc())
        return gt, f"{kind}:{model}", 0.0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["mle", "nuts"])
    ap.add_argument("--jobs", default="all", help="nuts: kind:model,...")
    ap.add_argument("--kinds", default=",".join(D.KINDS), help="mle: datasets")
    ap.add_argument("--truths", default=",".join(D.GTS))
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    sel = set(args.truths.split(","))
    lo, hi = (int(x) for x in args.seeds.split("-")) if "-" in args.seeds else (int(args.seeds),) * 2
    todo = []
    if args.kind == "mle":
        for seed in range(lo, hi + 1):
            for kind in args.kinds.split(","):
                for model in ("free", D.CONSTRAINED[kind]):
                    for gt in (g for g in D.gts(kind) if g in sel):
                        f = D.OUT / "mle" / f"{kind}_{model}" / f"{gt}_s{seed}"
                        if not (f.with_suffix(".json").exists()
                                or f.with_suffix(".lock").exists()):
                            todo.append((mle_job, (gt, seed, kind, model)))
    else:
        for km in args.jobs.split(","):
            kind, model = km.split(":")
            for gt in (g for g in D.gts("M" if kind == "M" else "D0") if g in sel):
                f = D.OUT / "nuts" / f"{kind}_{model}" / f"{gt}_s0"
                if not f.with_suffix(".json").exists() and not f.with_suffix(".lock").exists():
                    todo.append((nuts_job, (gt, kind, model, args.threads)))
    print(f"{len(todo)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(fn, *a) for fn, a in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            err, sec = r[-1], r[-2]
            print(f"[{k}/{len(todo)}] {time.time() - t0:.0f}s {r[:-2]} "
                  f"{'FAILED ' + err if err else f'{sec:.0f}s'}", flush=True)  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main()
