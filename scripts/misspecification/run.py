"""Misspecification study: job runner (restartable; finished outputs are skipped).

Jobs
----
``fit``    (scenario, truth, seed): simulate n = 512 detected clones per generation 4/6/8/10
           under the scenario; fit the standard model (A founder, perfect typing,
           homogeneous synchronous GW) by multistart MLE; binned G^2 goodness of fit;
           profile-likelihood 95% intervals of B.to_A and m_BA. Where the exact likelihood
           can represent the scenario: the same with the true setting known ("true") and
           the extended model with the setting estimated (LR test against the standard).
           Mis-typing datasets are the base dataset of the same seed, relabelled.
``dediff`` (scenario, null row, seed): the same data-generation under a no-dedifferentiation
           truth (N1, N2 of docs/dediff_test), LRT of H0: B.to_A = B.div_AA = B.div_AB = 0
           with chi-bar calibration, under the standard model (and the true setting where
           representable).

Usage::

    uv run python scripts/misspecification/run.py jobs [--seeds 0-4] [--dediff-seeds 0-49]
    uv run python scripts/misspecification/run.py run --workers 17 [--only fit|dediff]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C  # noqa: E402

N = 512
SCEN = list(C.SCENARIOS)
FIT_DIR = C.OUT / "fits"
DD_DIR = C.OUT / "dediff"
K_FILE = C.OUT / "density_K.json"


def density_Ks() -> dict:
    if K_FILE.exists():
        return json.loads(K_FILE.read_text())
    Ks = {g: C.density_K(P) for g, P in {**C.TRUTHS, **C.NULLS}.items()}
    C.OUT.mkdir(parents=True, exist_ok=True)
    K_FILE.write_text(json.dumps(Ks, indent=1))
    return Ks


def truth_P(name):
    return C.TRUTHS[name] if name in C.TRUTHS else C.NULLS[name]


def dataset(scen: str, gt: str, seed: int, Ks: dict):
    """Dataset of (scenario, truth, seed); mis-typing = base dataset relabelled."""
    P = truth_P(gt)
    gi = (C.GTS + list(C.NULLS)).index(gt)
    opts = C.SCENARIOS[scen][1]
    if "confusion" in opts:
        base = C.scenario_data("base", P, N, np.random.default_rng([gi, seed, 0]))
        conf = C.confusion_of(opts["confusion"])
        rng = np.random.default_rng([gi, seed, SCEN.index(scen), 1])
        from bio_sbi.summaries import Cohort

        return [Cohort(c.horizon, C.observe(c.counts, rng, 1.0, conf)) for c in base]
    return C.scenario_data(scen, P, N, np.random.default_rng([gi, seed, SCEN.index(scen)]),
                           K_lookup=Ks.get(gt))  # fmt: skip


def true_setting(scen: str) -> dict | None:
    """``set_model`` kwargs of the scenario's true law, if representable."""
    o = C.SCENARIOS[scen][1]
    if scen in ("F70", "D15A", "D15mix"):
        kw = {"founder_A": o.get("founder_A", 1.0)}
        if o.get("doublet"):
            kw["size_probs"] = (1 - o["doublet"], o["doublet"])
        return kw
    if "confusion" in o:
        return {"confusion": C.confusion_of(o["confusion"])}
    if "capture" in o:
        return {"capture": o["capture"]}
    return None


def _analyse(lik, rng, profiles: bool = True) -> dict:
    f = C.fit_mle(lik, rng)
    out = {"ll": f["ll"], "P": C.pdict(f["P"]), "summ": C.summaries(f["P"]),
           "start": f["start"], "n_tied": f["n_tied"], "second_ll": f["second_ll"],
           "gof": C.gof(lik, f["P"])}  # fmt: skip
    if profiles:
        out["profile"] = {fn: C.profile_interval(lik, f["P"], f["ll"], fn)
                          for fn in ("B.to_A", "B.m_BA")}  # fmt: skip
    return out, f


def job_fit(scen: str, gt: str, seed: int, Ks: dict) -> dict:
    t0 = time.perf_counter()
    data = dataset(scen, gt, seed, Ks)
    rng = np.random.default_rng([seed, 99])
    lik = C.make_lik(data)
    res = {"scen": scen, "gt": gt, "seed": seed, "truth": C.pdict(truth_P(gt)),
           "truth_summ": C.summaries(truth_P(gt)),
           "mean_cells": [float(c.counts.sum(1).mean()) for c in data]}  # fmt: skip
    prof = seed < 3  # profile intervals on datasets 0-2 only (budget)
    res["std"], f = _analyse(lik, rng, prof)
    ts = true_setting(scen)
    if ts is not None:
        C.set_model(lik, **ts)
        res["true"], _ = _analyse(lik, rng, prof)
        C.set_model(lik)
    # extended models (and their size on the base data): job_ext, grid method
    res["seconds"] = time.perf_counter() - t0
    return res


def job_dediff(scen: str, null: str, seed: int, Ks: dict) -> dict:
    t0 = time.perf_counter()
    data = dataset(scen, null, seed + 1000, Ks)
    res = {"scen": scen, "null": null, "seed": seed,
           "mean_cells": [float(c.counts.sum(1).mean()) for c in data]}  # fmt: skip
    lik = C.make_lik(data)
    variants = [("std", {})]
    ts = true_setting(scen)
    if ts is not None:
        variants.append(("true", ts))
    for name, kw in variants:
        C.set_model(lik, **kw)
        r = C.dl.run_test(lik, np.random.default_rng([seed, 7]), chibar_seed=seed)
        res[name] = {k: r[k] for k in ("lambda", "m_ba_hat", "q_hat", "p_chibar", "p_chi2_3",
                                       "p_binomial", "ll0", "ll1", "theta1", "theta0")}
        res[name]["crit95"] = r["chibar"]["crit95"]
    res["seconds"] = time.perf_counter() - t0
    return res


EXT_DIR = C.OUT / "fits_ext"


def job_ext(scen: str, gt: str, seed: int, Ks: dict) -> dict:
    """Extended model by the grid method (``C.fit_extended_grid``), from the fit job's
    standard MLE of the same dataset."""
    t0 = time.perf_counter()
    fit = json.loads((FIT_DIR / scen / gt / f"s{seed}.json").read_text())
    data = dataset(scen, gt, seed, Ks)
    lik = C.make_lik(data)
    P = np.array([fit["std"]["P"][n] for n in C.NAMES]).reshape(2, 6)
    s = C.FULL.encode(P)
    std = {"P": P, "s": s, "ll": float(C.dl.loglik(lik, C.FULL, s, False)[0])}
    rng = np.random.default_rng([seed, 98])
    res = {"scen": scen, "gt": gt, "seed": seed}
    if scen == "base" and seed < 3:  # size of the extended-model LR tests (base data)
        res["ext_null"] = {}
        for kind in ("fA", "d2", "eBA", "cap"):
            e = C.fit_extended_grid(lik, kind, std, rng)
            res["ext_null"][kind] = {"lr": e["lr"], "p": C.lrt_pvalue(e["lr"], kind),
                                     "x": e["x"]}
    for kind in C.SCENARIOS[scen][2]:
        if kind == "true":
            continue
        e = C.fit_extended_grid(lik, kind, std, rng)
        res["ext"] = {"kind": kind, "ll": e["ll"], "P": C.pdict(e["P"]), "x": e["x"],
                      "summ": C.summaries(e["P"]), "lr": e["lr"], "grid": e["grid"],
                      "std_ll_used": e["std_ll_used"], "p": C.lrt_pvalue(e["lr"], kind)}
    res["seconds"] = time.perf_counter() - t0
    return res


def out_path(job) -> Path:
    kind, scen, gt, seed = job
    d = {"fit": FIT_DIR, "dediff": DD_DIR, "ext": EXT_DIR}[kind]
    return d / scen / gt / f"s{seed}.json"


def _run(job, Ks):
    kind, scen, gt, seed = job
    p = out_path(job)
    try:
        res = {"fit": job_fit, "dediff": job_dediff, "ext": job_ext}[kind](scen, gt, seed, Ks)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, default=float))
        tmp.rename(p)
        return job, res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".err").write_text(traceback.format_exc())
        return job, 0.0, repr(exc)


def parse_range(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def make_jobs(seeds, dd_seeds, scen=None):
    scen = scen or SCEN
    jobs = [("fit", s, g, k) for k in seeds for s in scen for g in C.GTS]
    jobs += [("dediff", s, nl, k) for k in dd_seeds for s in scen for nl in C.NULLS]
    jobs += [("ext", s, g, k) for k in seeds for s in scen for g in C.GTS
             if (s == "base" and k < 3) or [x for x in C.SCENARIOS[s][2] if x != "true"]]
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "one"])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--dediff-seeds", default="0-49")
    ap.add_argument("--scen", default="")
    ap.add_argument("--workers", type=int, default=17)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    Ks = density_Ks()
    scen = args.scen.split(",") if args.scen else None
    seeds = parse_range(args.seeds) if args.seeds else []
    dd = parse_range(args.dediff_seeds) if args.dediff_seeds else []
    jobs = make_jobs(seeds, dd, scen)
    if args.only:
        jobs = [j for j in jobs if j[0] == args.only]
    todo = [j for j in jobs if not out_path(j).exists()
            and (j[0] != "ext" or out_path(("fit",) + j[1:]).exists())]
    print(f"{len(todo)} / {len(jobs)} jobs to run, {args.workers} workers", flush=True)
    if args.cmd == "one":
        print(json.dumps(_run(todo[0], Ks)[1:], default=float))
        return
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(_run, j, Ks) for j in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            job, sec, err = fu.result()
            msg = f"FAILED {err}" if err else f"{sec:.0f}s"
            print(f"[{k}/{len(todo)}] {time.time() - t0:.0f}s {'/'.join(map(str, job))} {msg}",
                  flush=True)  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main()
