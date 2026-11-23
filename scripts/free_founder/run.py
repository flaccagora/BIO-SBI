"""Free founder distribution in the exact GW NUTS pipeline: misspecification founder scenarios.

Data: the founder scenarios of the misspecification study (``scripts/misspecification``:
same simulator, seeds and datasets), n = 512 detected clones per generation 4/6/8/10:
``base`` (one A founder), ``F70`` (founder A/B 70/30), ``D15A`` (15% doublets, A),
``D15mix`` (15% doublets, founders 70/30); the 9 XG2 truths. The simulator is replayed
here with the true founders of every clone recorded (the counts are checked to be
identical to ``run.dataset`` of the misspecification study).

Models (``inference=exact_gw`` with free founder rows, flat priors):
``fA``    = founder sizes {1}, ``founder.A`` free (Uniform(0, 1));
``fA+d2`` = founder sizes {1, 2}, ``founder.A`` and ``founder_size.2`` free (both Uniform).

NUTS as the misspecification study's all-A runs (adjoint kernel, 4 chains from L-BFGS optima
of 8 prior draws, 300 warmup, max tree depth 7, stop at rank R-hat <= 1.01, bulk ESS >= 200,
tail ESS >= 100, chain-mean spread <= 0.5, or 500 draws per chain). Per-clone progenitor
posterior P(founders = c | counts) averaged over 100 posterior draws (and at the posterior
mean), saved with the true founders.

    uv run python scripts/free_founder/run.py --jobs D15mix:fA+d2,F70:fA --seeds 0 \
        --workers 4 --threads 4 [--out DIR]

Outputs per job ``<out>/<scen>/<model>/<gt>_s<seed>.{json,npz}`` (restartable: finished
or locked jobs are skipped).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "misspecification"))
import common as C  # noqa: E402

OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/free_founder")
N = 512
SCEN = list(C.SCENARIOS)  # seed stream index as in the misspecification study
MODELS = {"fA": [1], "fA+d2": [1, 2]}
SUMMARIES = ("rho", "stableA", "m_BA", "B.div")
N_PROG_DRAWS = 100


# ---------------------------------------------------------------------- data
def _simulate_true(P, gens, n, rng, founder_A=1.0, doublet=0.0):
    """``common.simulate_true`` (founder / doublet options only), also returning founders."""
    P = np.asarray(P, float)
    size = 1 + (rng.random(n) < doublet)
    nA0 = rng.binomial(size, founder_A)
    a, b = nA0.astype(np.int64), (size - nA0).astype(np.int64)
    for _ in range(1, gens + 1):
        a, b = C._step(a, b, P, rng)
        if a.max(initial=0) + b.max(initial=0) > 200_000:
            raise RuntimeError("max_cells exceeded")
    return np.stack([a, b], 1), np.stack([size, nA0], 1)


def dataset_with_founders(scen: str, gt: str, seed: int):
    """Cohorts of ``run.dataset(scen, gt, seed)`` and the true (founder count, A founders)."""
    from bio_sbi.summaries import Cohort

    o = C.SCENARIOS[scen][1]
    assert set(o) <= {"founder_A", "doublet"}, scen
    P = C.TRUTHS[gt]
    gi = (C.GTS + list(C.NULLS)).index(gt)
    rng = np.random.default_rng([gi, seed, SCEN.index(scen)])
    fa, d2 = o.get("founder_A", 1.0), o.get("doublet", 0.0)
    cohorts, founders = [], []
    for h in C.HORIZONS:  # = common.simulate (capture 1, no confusion: observe is identity)
        kept, kf, have, batch = [], [], 0, int(1.5 * N) + 64
        while have < N:
            tr, fo = _simulate_true(P, h, batch, rng, fa, d2)
            ob = C.observe(tr, rng, 1.0, None)
            det = ob.sum(1) > 0
            kept.append(ob[det])
            kf.append(fo[det])
            have += int(det.sum())
            rate = max(have, 1) / (batch * len(kept))
            batch = int(min(max((N - have) / rate * 1.3, 64), 50 * N))
        cohorts.append(Cohort(float(h), np.concatenate(kept)[:N]))
        founders.append(np.concatenate(kf)[:N])
    return cohorts, founders


def check_dataset(scen: str, gt: str, seed: int, cohorts) -> None:
    """Check the counts against the misspecification study's dataset of the same seed."""
    ref = C.scenario_data(scen, C.TRUTHS[gt], N, np.random.default_rng(
        [(C.GTS + list(C.NULLS)).index(gt), seed, SCEN.index(scen)]))  # fmt: skip
    for a, b in zip(cohorts, ref, strict=True):
        if not np.array_equal(a.counts, b.counts):
            raise RuntimeError(f"{scen}/{gt}/s{seed}: replayed data differ from the study's")


# ---------------------------------------------------------------------- model
def make_sim(sizes):
    import yaml

    from bio_sbi.simulators import GaltonWatson

    conf = ROOT / "src" / "bio_sbi" / "conf" / "law" / "two_type.yaml"
    law = yaml.safe_load(conf.read_text())
    law.update(fixed={}, founder_sizes=list(sizes))
    return GaltonWatson(law)


def physical_names(sim) -> list[str]:
    return list(sim.space.names)


def stats_of(v: np.ndarray, truth: float | None) -> dict:
    mu, sd = float(v.mean()), float(v.std())
    q05, q50, q95 = (float(x) for x in np.quantile(v, [0.05, 0.5, 0.95]))
    out = {"mean": mu, "sd": sd, "q05": q05, "median": q50, "q95": q95}
    if truth is not None:
        out.update(truth=float(truth), z=(mu - truth) / sd if sd > 0 else None,
                   cover90=bool(q05 <= truth <= q95))  # fmt: skip
    return out


def job(scen: str, model: str, gt: str, seed: int, threads: int, out_dir: Path) -> tuple:
    from bio_sbi.inference import dediff_lrt as dl
    from bio_sbi.inference.exact_gw import ExactGW, ExactLikelihood

    t0 = time.perf_counter()
    out = out_dir / scen / model / f"{gt}_s{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out.with_suffix(".lock"), os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return scen, model, gt, seed, 0.0, "locked"
    try:
        cohorts, founders = dataset_with_founders(scen, gt, seed)
        check_dataset(scen, gt, seed, cohorts)
        sim = make_sim(MODELS[model])
        lik = ExactLikelihood(sim, dl.observation(None), cohorts, kernel="adjoint")
        m = ExactGW(kernel="adjoint", threads=threads, chains=4, init_starts=8, warmup=300,
                    draws=500, log_every=0, seed=seed, max_tree_depth=7,
                    stop={"enabled": True, "ess": 200.0, "tail_ess": 100.0,
                          "max_draws": 500})  # fmt: skip
        theta, diag = m.run(lik, key=("free_founder", scen, model, gt, seed))
        t_nuts = time.perf_counter() - t0
        names = physical_names(sim)
        phys = sim.space.decode(theta)
        P12 = phys[:, :12].reshape(-1, 2, 6)
        summ = [C.summaries(P) for P in P12]
        cols = {n: phys[:, i] for i, n in enumerate(names)}
        for k in SUMMARIES:
            cols[k] = np.array([s[k] for s in summ])
        o = C.SCENARIOS[scen][1]
        truth = {**C.pdict(C.TRUTHS[gt]), **C.summaries(C.TRUTHS[gt]),
                 "founder.A": o.get("founder_A", 1.0), "founder.B": 1 - o.get("founder_A", 1.0),
                 "founder_size.1": 1 - o.get("doublet", 0.0),
                 "founder_size.2": o.get("doublet", 0.0)}  # fmt: skip
        res = {"scen": scen, "model": model, "gt": gt, "seed": seed, "draws": int(len(phys)),
               "stats": {k: stats_of(v, truth[k]) for k, v in cols.items() if k in truth}}
        # correlation of p_A with the B row and with B->A (weak "parent swap" directions)
        res["corr"] = {f"founder.A~{k}": float(np.corrcoef(cols["founder.A"], cols[k])[0, 1])
                       for k in ("B.to_A", "m_BA", "B.div", "B.death", "B.persist")}  # fmt: skip
        if "founder_size.2" in cols:
            res["corr"].update({f"founder_size.2~{k}": float(np.corrcoef(
                cols["founder_size.2"], cols[k])[0, 1]) for k in ("founder.A", "B.to_A", "rho")})  # fmt: skip
        keep = ("max_rank_rhat", "min_ess_bulk", "min_ess_tail", "max_mean_spread", "divergences",
                "warmup_divergences", "max_tree_depth_hits", "draws_per_chain", "step_size",
                "init_log_posterior", "chain_mean_log_posterior", "seconds", "seconds_per_eval",
                "gradient_evals", "chain_means")  # fmt: skip
        res["convergence"] = {k: diag.get(k) for k in keep}
        res["stop"] = {k: diag["stop"][k] for k in ("reason", "failed", "draws_per_chain")}
        res["seconds_nuts"] = t_nuts
        # ---- per-clone progenitor posterior
        t1 = time.perf_counter()
        rng = np.random.default_rng([seed, 11])
        idx = rng.choice(len(theta), size=min(N_PROG_DRAWS, len(theta)), replace=False)
        labels, acc = None, None
        for i in idx:
            labels, probs = lik.progenitor_posterior(theta[i])
            acc = probs if acc is None else [x + p for x, p in zip(acc, probs, strict=True)]
        avg = [x / len(idx) for x in acc]
        _, at_mean = lik.progenitor_posterior(theta.mean(0))
        per_clone, per_clone_mean, info = [], [], []
        for h, c, f, pa, pm in zip(lik.data, cohorts, founders, avg, at_mean, strict=True):
            cell = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(h.a, h.b, strict=True))}
            j = np.array([cell[(int(a), int(b))] for a, b in c.counts])
            per_clone.append(pa[j])
            per_clone_mean.append(pm[j])
            info.append(np.column_stack([np.full(len(j), h.generations), c.counts, f]))
        res["progenitor_labels"] = labels
        res["seconds_progenitor"] = time.perf_counter() - t1
        np.savez_compressed(
            out.with_suffix(".npz"), phys=phys.astype(np.float32), names=np.array(names),
            chain=np.repeat(np.arange(4), len(phys) // 4).astype(np.int8),
            clones=np.concatenate(info).astype(np.int32),  # generation, A, B, size, n_A founders
            prog=np.concatenate(per_clone).astype(np.float32),
            prog_at_mean=np.concatenate(per_clone_mean).astype(np.float32),
            labels=np.array(labels),
        )  # fmt: skip
        res["seconds"] = time.perf_counter() - t0
        out.write_text(json.dumps(res, default=float, indent=1))
        return scen, model, gt, seed, res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".err").write_text(traceback.format_exc())
        return scen, model, gt, seed, 0.0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="D15mix:fA+d2,F70:fA,D15A:fA+d2,base:fA,F70:fA+d2,"
                                      "base:fA+d2")  # fmt: skip
    ap.add_argument("--gts", default=",".join(C.GTS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out)
    seeds = [int(s) for s in args.seeds.split(",")]
    todo = []
    for seed in seeds:
        for sm in args.jobs.split(","):
            scen, model = sm.split(":")
            for gt in args.gts.split(","):
                f = out_dir / scen / model / f"{gt}_s{seed}"
                if not f.with_suffix(".json").exists() and not f.with_suffix(".lock").exists():
                    todo.append((scen, model, gt, seed))
    print(f"{len(todo)} jobs", flush=True)
    if args.dry:
        for t in todo:
            print(*t)
        return
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(job, *t, args.threads, out_dir) for t in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            s, mo, g, sd, sec, err = fut.result()
            print(f"[{k}/{len(todo)}] {time.time() - t0:.0f}s {s}/{mo}/{g}/s{sd} "
                  f"{'FAILED ' + err if err else f'{sec:.0f}s'}", flush=True)  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main()
