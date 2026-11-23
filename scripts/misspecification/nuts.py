"""Flat-prior exact NUTS posteriors of the standard model on misspecified datasets.

One dataset (seed 0, the same dataset as the MLE job of that seed) per truth and scenario;
4 chains, adjoint kernel, adaptive stopping (R-hat / ESS targets). Writes per run
``nuts/<scen>/<gt>.json`` (posterior mean, sd, 5/95% quantiles, truth, z, 90% coverage for
the 12 probabilities and the summaries rho, stableA, m_BA, B.div) and ``.npz`` draws.

    uv run python scripts/misspecification/nuts.py --scen MTba2,F70,HET10 --workers 4
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402
import run as R  # noqa: E402

NUTS_DIR = C.OUT / "nuts"


def one(scen: str, gt: str, threads: int) -> tuple:
    from bio_sbi.inference.exact_gw import ExactGW

    t0 = time.perf_counter()
    out = NUTS_DIR / scen / f"{gt}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:  # lock: several nuts.py processes may share the job list
        os.close(os.open(out.with_suffix(".lock"), os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return scen, gt, 0.0, "locked (running elsewhere)"
    try:
        Ks = R.density_Ks()
        data = R.dataset(scen, gt, 0, Ks)
        lik = C.make_lik(data)
        # reduced run length (budget): 300 warmup, stop at bulk ESS 200 / tail ESS 100 /
        # R-hat 1.01 or 500 draws per chain
        m = ExactGW(kernel="adjoint", threads=threads, chains=4, init_starts=8, warmup=300,
                    draws=500, log_every=0, seed=0, max_tree_depth=7,
                    stop={"enabled": True, "ess": 200.0, "tail_ess": 100.0,
                          "max_draws": 500})  # fmt: skip
        theta, diag = m.run(lik, key=("missp", scen, gt))
        phys = C.sim().decode(np.asarray(theta))[:, :12]
        Ps = phys.reshape(-1, 2, 6)
        summ = [C.summaries(P) for P in Ps]
        cols = {n: phys[:, i] for i, n in enumerate(C.NAMES)}
        for k in summ[0]:
            cols[k] = np.array([s[k] for s in summ])
        truth = {**C.pdict(C.TRUTHS[gt]), **C.summaries(C.TRUTHS[gt])}
        res = {"scen": scen, "gt": gt, "stats": {}}
        for k, v in cols.items():
            mu, sd = float(v.mean()), float(v.std())
            q05, q95 = (float(x) for x in np.quantile(v, [0.05, 0.95]))
            t = truth[k]
            res["stats"][k] = {"mean": mu, "sd": sd, "q05": q05, "q95": q95, "truth": t,
                               "z": (mu - t) / sd if sd > 0 else None,
                               "cover90": bool(q05 <= t <= q95)}  # fmt: skip
        res["draws"] = int(len(phys))
        res["convergence"] = {"max_rhat": diag.get("max_rank_rhat"),
                              "min_ess_bulk": diag.get("min_ess_bulk"),
                              "min_ess_tail": diag.get("min_ess_tail"),
                              "max_mean_spread": diag.get("max_mean_spread"),
                              "divergences": diag.get("divergences"),
                              "max_tree_depth_hits": diag.get("max_tree_depth_hits")}
        res["stop"] = diag.get("stop")
        res["seconds"] = time.perf_counter() - t0
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix(".npz"), phys=phys)
        out.write_text(json.dumps(res, default=str, indent=1))
        return scen, gt, res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".err").write_text(traceback.format_exc())
        return scen, gt, 0.0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="MTba2,F70,HET10")
    ap.add_argument("--gts", default=",".join(C.GTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    jobs = [(s, g) for s in args.scen.split(",") for g in args.gts.split(",")
            if not (NUTS_DIR / s / f"{g}.json").exists()
            and not (NUTS_DIR / s / f"{g}.lock").exists()]  # fmt: skip
    print(f"{len(jobs)} NUTS runs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(one, s, g, args.threads) for s, g in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            s, g, sec, err = f.result()
            print(f"[{k}/{len(jobs)}] {time.time() - t0:.0f}s {s}/{g} "
                  f"{'FAILED ' + err if err else f'{sec:.0f}s'}", flush=True)  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main()
