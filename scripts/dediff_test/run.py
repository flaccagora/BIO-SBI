"""Likelihood-ratio test for global dedifferentiation (B -> A): simulation studies and data.

H0: B.to_A = B.div_AA = B.div_AB = 0 against the full two-type GW model (10 free
probabilities), exact likelihood of the endpoint counts (src/bio_sbi/inference/dediff_lrt.py).

Studies (one JSON per dataset under results/exact_gw/dediff_test/<study>/<cell>/):

  null    N1..N4 (cells.py), perfect typing; every dataset analysed at n = 128, 256,
          512, 1024 clones per endpoint (generations 4, 6, 8, 10), nested: the first n
          clones of the n = 1024 dataset; s<seed>.json
  alt     N1/N2 x route (toA, divAB, mixed) x m_BA (0.02 ... 0.4)
  typing  N1/N2 data mis-typed (symmetric 2% / 5%; B read as A 0.5% / 1% / 2% / 5%),
          analysed assuming perfect typing ("ignored"), with the known confusion matrix
          ("modelled") and with half the true rate ("half"); the clones are those of the
          null datasets with the same seed
  ratio   how precisely the error rate must be known: data mis-typed at rate e (cells
          N1_bA0.02 under H0, N1_mixed_m0.01_bA0.02 under H1; cells.py RATIOS ...)
          analysed with the confusion matrix at the assumed rate r * e for every r in
          --ratios (analyses "r0.5", "r1.1", ...); same clones as the null / alt datasets
          with the same seed; results/.../typing_ratio/<cell>/s<seed>.json (resumable per
          analysis: missing ratios are added to an existing file)
  boot    per-dataset parametric bootstrap of null datasets (simulate from theta_0,
          refit both models); n<n>_s<seed>.json
  data    a measured CSV (columns horizon,A,B; horizons in generations): LRT, asymptotic
          p-values, parametric bootstrap p-value

  uv run python scripts/dediff_test/run.py null --seeds 0:200 --workers 16
  uv run python scripts/dediff_test/run.py alt --seeds 0:50 --m 0.0025,0.005,0.01,0.02,0.05
  uv run python scripts/dediff_test/run.py alt --seeds 0:10 --m 0.1,0.2,0.4
  uv run python scripts/dediff_test/run.py typing --seeds 0:100
  uv run python scripts/dediff_test/run.py boot --cells N1,N2,N3,N4 --ns 256,1024 \
      --seeds 0:10 --reps 100
  uv run python scripts/dediff_test/run.py data data/real/crc0322_2d_gw.csv --reps 200 \
      [--confusion 0.02 --typing-kind b_as_a] [--out DIR]
  uv run python scripts/dediff_test/run.py data data/real/crc0322_2d_gw.csv --reps 200 \
      --confusion-interval 0.015 0.025 [--interval-points 5]   # uncertain error rate
  uv run python scripts/dediff_test/run.py ratio --cells ratio_null --seeds 0:100
  uv run python scripts/dediff_test/run.py ratio --cells ratio_alt --seeds 0:50

Datasets: seed s uses the pipeline's reference stream (data_seed = s), so N1 seed 3 is
the reference dataset of `data_seed=3` with this truth. Existing outputs are skipped
unless --force. One thread per worker (OMP/NUMBA/MKL = 1, set here).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cells as C  # noqa: E402

from bio_sbi.inference import dediff_lrt as D  # noqa: E402
from bio_sbi.rng import rng_for, seed_for  # noqa: E402

ROOT = Path("/home/matteo/BIO-SBI/results/exact_gw/dediff_test")

SETTINGS = {
    # pilot: every start family, generously
    "rich": D.FitSettings(h0_random=6, full_random=4, q_starts=(0.01, 0.1),
                          pins=(0.02, 0.1, 0.3, 0.6)),  # fmt: skip
    # main studies (chosen from the pilot, see docs/dediff_test/README.md)
    "default": D.FitSettings(h0_random=3, full_random=1, q_starts=(0.01,),
                             pins=(0.02, 0.3)),  # fmt: skip
    # bootstrap replicates (+ the generating theta_0 as an H0 start)
    "boot": D.FitSettings(h0_random=2, full_random=1, q_starts=(0.01,), pins=(0.02, 0.3)),
}


def _json(obj):
    if isinstance(obj, dict):
        return {str(k): _json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(_json(payload), indent=1))
    tmp.replace(path)


def _counts_summary(cohorts) -> list[dict]:
    return [{"horizon": c.horizon, "n": len(c.counts), "mean_A": float(c.counts[:, 0].mean()),
             "mean_B": float(c.counts[:, 1].mean()), "max_A": int(c.counts[:, 0].max()),
             "max_B": int(c.counts[:, 1].max()), "frac_A0": float(np.mean(c.counts[:, 0] == 0))}
            for c in cohorts]  # fmt: skip


def _truth_check(lik, P: np.ndarray) -> dict:
    """Both models maximised from the truth (diagnostic only; never used in Lambda)."""
    H0, FULL = D.h0_model(), D.full_model()
    r0 = D.maximise(lik, H0, H0.encode(P))
    r1 = D.maximise(lik, FULL, FULL.encode(P, {1: np.ones(3)}))
    return {"ll0": r0["ll"], "ll1": r1["ll"]}


# ---------------------------------------------------------------------- simulation task
def sim_task(study: str, name: str, seed: int, settings: str, ns: tuple, out: str,
             truth_check: bool) -> dict:  # fmt: skip
    t0 = time.perf_counter()
    sim = D.make_simulator()
    c = C.cell(name)
    data = D.simulate_counts(sim, c["P"], C.HORIZONS, max(ns), seed, ("reference",),
                             c["confusion"])  # fmt: skip
    # typing cells: confusion ignored, modelled (known rate), modelled with half the rate
    analyses = {"perfect": None} if c["typing"] is None else \
        {"ignored": None, "modelled": c["confusion"], "half": c["confusion_half"]}  # fmt: skip
    results = {}
    for an, conf in analyses.items():
        warm, recs = None, []
        for n in ns:
            sub = D.subset(data, n)
            lik = D.likelihood(sim, sub, conf)
            rng = rng_for(seed, "starts", name, an, n)
            rec = D.run_test(lik, rng, SETTINGS[settings], warm,
                             chibar_seed=seed_for(seed, "chibar", name, an, n))  # fmt: skip
            rec["n"] = n
            if truth_check:
                tc = _truth_check(lik, c["P"])
                rec["truth_ll0"], rec["truth_ll1"] = tc["ll0"], tc["ll1"]
                rec["missed0"] = tc["ll0"] - rec["ll0"]   # > 0: H0 fit missed its optimum
                rec["missed1"] = tc["ll1"] - rec["ll1"]
            rec["data"] = _counts_summary(sub)
            recs.append(rec)
            warm = {"h0": [np.array(rec["P0"])], "full": [np.array(rec["P1"])]}
        results[an] = recs
    payload = {"study": study, "cell": name, "seed": seed, "settings": settings,
               "truth": D.P_to_dict(c["P"]), "m_ba": c["m"], "route": c["route"],
               "base": c["base"], "typing": c["typing"], "confusion": c["confusion"],
               "horizons": list(C.HORIZONS), "results": results,
               "seconds": time.perf_counter() - t0}  # fmt: skip
    _write(Path(out), payload)
    return {"out": out, "seconds": payload["seconds"],
            "lambda": [r["lambda"] for r in next(iter(results.values()))]}  # fmt: skip


# ---------------------------------------------------------------------- ratio task
def ratio_key(r: float) -> str:
    """Analysis name of the assumed-rate ratio ``r``."""
    return f"r{r:g}"


def ratio_task(name: str, seed: int, ratios: tuple, settings: str, ns: tuple, out: str,
               truth_check: bool) -> dict:  # fmt: skip
    """Mis-typed dataset (name, seed) analysed at assumed rates r * e, r in ``ratios``.

    Every ratio is a chain over ``ns`` (warm starts from the next smaller n, as in
    sim_task). Analyses already in ``out`` are kept; the file is rewritten after each
    ratio, so an interrupted task resumes where it stopped.
    """
    t0 = time.perf_counter()
    sim = D.make_simulator()
    c = C.cell(name)
    if c["typing"] is None:
        raise ValueError(f"{name}: the ratio study needs mis-typed data")
    path = Path(out)
    old = json.loads(path.read_text()) if path.exists() else {}
    results = dict(old.get("results", {}))
    data = D.simulate_counts(sim, c["P"], C.HORIZONS, max(ns), seed, ("reference",),
                             c["confusion"])  # fmt: skip
    for ratio in ratios:
        an = ratio_key(ratio)
        if an in results and {r["n"] for r in results[an]} >= set(ns):
            continue
        rate = ratio * c["rate"]
        conf = D.confusion_matrix(rate, c["kind"])
        warm, recs = None, []
        for n in ns:
            sub = D.subset(data, n)
            lik = D.likelihood(sim, sub, conf)
            rng = rng_for(seed, "starts", name, an, n)
            rec = D.run_test(lik, rng, SETTINGS[settings], warm,
                             chibar_seed=seed_for(seed, "chibar", name, an, n))  # fmt: skip
            rec.update(n=n, ratio=ratio, assumed_rate=rate)
            if truth_check and ratio == 1.0:
                tc = _truth_check(lik, c["P"])
                rec["truth_ll0"], rec["truth_ll1"] = tc["ll0"], tc["ll1"]
                rec["missed0"] = tc["ll0"] - rec["ll0"]
                rec["missed1"] = tc["ll1"] - rec["ll1"]
            rec["data"] = _counts_summary(sub)
            recs.append(rec)
            warm = {"h0": [np.array(rec["P0"])], "full": [np.array(rec["P1"])]}
        results[an] = recs
        payload = {"study": "ratio", "cell": name, "seed": seed, "settings": settings,
                   "truth": D.P_to_dict(c["P"]), "m_ba": c["m"], "route": c["route"],
                   "base": c["base"], "typing": c["typing"], "rate": c["rate"],
                   "kind": c["kind"], "confusion": c["confusion"],
                   "horizons": list(C.HORIZONS), "results": results,
                   "seconds": old.get("seconds", 0.0) + time.perf_counter() - t0}  # fmt: skip
        _write(path, payload)
    lam = {an: recs[-1]["lambda"] for an, recs in results.items()}
    return {"out": out, "seconds": time.perf_counter() - t0,
            "lambda": [lam[ratio_key(r)] for r in ratios if ratio_key(r) in lam]}  # fmt: skip


def ratio_done(path: Path, ratios: tuple, ns: tuple) -> bool:
    """True if ``path`` holds every ratio analysis at every n."""
    if not path.exists():
        return False
    res = json.loads(path.read_text()).get("results", {})
    return all(ratio_key(r) in res and {x["n"] for x in res[ratio_key(r)]} >= set(ns)
               for r in ratios)  # fmt: skip


# ---------------------------------------------------------------------- bootstrap task
def boot_task(name: str, seed: int, n: int, reps: int, out: str, null_file: str) -> dict:
    """Parametric bootstrap of the null dataset (name, seed) at n clones per endpoint."""
    t0 = time.perf_counter()
    sim = D.make_simulator()
    c = C.cell(name)
    src = Path(null_file)
    if src.exists():  # the original analysis of the null study (same Lambda, theta_0)
        rec = next(r for r in json.loads(src.read_text())["results"]["perfect"] if r["n"] == n)
    else:
        data = D.simulate_counts(sim, c["P"], C.HORIZONS, n, seed, ("reference",))
        lik = D.likelihood(sim, data)
        rec = D.run_test(lik, rng_for(seed, "starts", name, "perfect", n), SETTINGS["default"])
    P0 = np.array(rec["P0"])
    reps_out = D.parametric_bootstrap(sim, P0, C.HORIZONS, n, range(reps), seed, (name, n),
                                      settings=SETTINGS["boot"])  # fmt: skip
    lam_b = np.array([r["lambda"] for r in reps_out])
    payload = {"cell": name, "seed": seed, "n": n, "reps": reps, "lambda": rec["lambda"],
               "P0": P0, "chibar": rec.get("chibar"), "p_chibar": rec.get("p_chibar"),
               "p_boot": float((1 + np.sum(lam_b >= rec["lambda"])) / (1 + reps)),
               "crit95_boot": float(np.quantile(lam_b, 0.95)), "boot": reps_out,
               "seconds": time.perf_counter() - t0}  # fmt: skip
    _write(Path(out), payload)
    return {"out": out, "seconds": payload["seconds"], "p_boot": payload["p_boot"]}


# ---------------------------------------------------------------------- data mode
def _read_csv(path: Path):
    import pandas as pd

    from bio_sbi.summaries import Cohort

    t = pd.read_csv(path)
    missing = {"horizon", "A", "B"} - set(t.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    cohorts = []
    for h in sorted(t["horizon"].unique()):
        x = t[t["horizon"] == h][["A", "B"]].to_numpy(dtype=np.int64)
        x = x[x.sum(1) > 0]  # detected clones only
        cohorts.append(Cohort(float(h), x))
    return cohorts


def _boot_rep(P0, horizons, ns, conf, seed, b) -> dict:
    sim = D.make_simulator()
    return D.parametric_bootstrap(sim, np.array(P0), horizons, ns, [b], seed, ("data",),
                                  conf, SETTINGS["boot"])[0]  # fmt: skip


def data_interval_mode(args) -> None:
    """Data mode with an uncertain error rate: test at a grid over the interval, keep the
    most conservative result (largest p-value)."""
    path = Path(args.csv)
    cohorts = _read_csv(path)
    horizons = [int(c.horizon) for c in cohorts]
    ns = [len(c.counts) for c in cohorts]
    lo, hi = args.confusion_interval
    rates = D.confusion_grid(lo, hi, args.interval_points)
    out = Path(args.out) if args.out else ROOT / "data" / f"{path.stem}_interval"
    sim = D.make_simulator()
    print(f"{path}: horizons {horizons}, clones {ns}; assumed {args.typing_kind} rates "
          f"{', '.join(f'{r:.4g}' for r in rates)}", flush=True)  # fmt: skip
    res = D.robust_test(sim, cohorts, rates, args.typing_kind, args.seed, (path.stem,),
                        SETTINGS[args.data_settings])  # fmt: skip
    for rec in res["per_rate"]:
        print(f"  rate {rec['rate']:.4g}: Lambda = {rec['lambda']:.3f} (m_BA_hat "
              f"{rec['m_ba_hat']:.4f}); p chi-bar {rec['p_chibar']:.4f}, chi-bar 95% "
              f"critical value {rec['chibar']['crit95']:.3f}", flush=True)  # fmt: skip
        if args.reps:
            conf = D.confusion_matrix(rec["rate"], args.typing_kind)
            with ProcessPoolExecutor(args.workers) as pool:
                futs = [pool.submit(_boot_rep, rec["P0"], horizons, ns, conf, args.seed, b)
                        for b in range(args.reps)]  # fmt: skip
                lam_b = np.array([f.result()["lambda"] for f in futs])
            rec["p_boot"] = float((1 + np.sum(lam_b >= rec["lambda"])) / (1 + len(lam_b)))
            rec["crit95_boot"] = float(np.quantile(lam_b, 0.95))
            rec["boot_lambda"] = lam_b
            print(f"    bootstrap ({args.reps} reps): p = {rec['p_boot']:.4f}, 95% critical "
                  f"value {rec['crit95_boot']:.3f}", flush=True)  # fmt: skip
    rob = D.robust_combine(res["per_rate"], rates)
    msg = (f"most conservative over [{lo:g}, {hi:g}]: smallest Lambda {rob['lambda_min']:.3f} "
           f"(rate {rob['rate_lambda_min']:.4g}); largest p chi-bar {rob['p_chibar']:.4f} "
           f"(rate {rob['rate_p_chibar']:.4g})")  # fmt: skip
    if "p_boot" in rob:
        msg += f"; largest bootstrap p {rob['p_boot']:.4f} (rate {rob['rate_p_boot']:.4g})"
    print(msg, flush=True)
    payload = {"csv": str(path), "horizons": horizons, "clones": ns, "kind": args.typing_kind,
               "interval": [lo, hi], "rates": rates, "robust": rob,
               "per_rate": res["per_rate"], "data": _counts_summary(cohorts)}  # fmt: skip
    _write(out / "result.json", payload)
    print(f"wrote {out / 'result.json'}")


def data_mode(args) -> None:
    if args.confusion_interval:
        data_interval_mode(args)
        return
    path = Path(args.csv)
    cohorts = _read_csv(path)
    horizons = [int(c.horizon) for c in cohorts]
    ns = [len(c.counts) for c in cohorts]
    conf = None if not args.confusion else D.confusion_matrix(args.confusion, args.typing_kind)
    out = Path(args.out) if args.out else ROOT / "data" / path.stem
    sim = D.make_simulator()
    lik = D.likelihood(sim, cohorts, conf)
    print(f"{path}: horizons {horizons}, clones {ns}; grids "
          f"{[(g.Lx, g.Ly) for g in lik.grids]}", flush=True)  # fmt: skip
    rec = D.run_test(lik, rng_for(args.seed, "starts", path.stem),
                     SETTINGS[args.data_settings])  # fmt: skip
    print(f"Lambda = {rec['lambda']:.3f}  (m_BA_hat {rec['m_ba_hat']:.4f}); p chi-bar "
          f"{rec['p_chibar']:.4f}, binomial {rec['p_binomial']:.4f}, chi2_3 "
          f"{rec['p_chi2_3']:.4f}", flush=True)  # fmt: skip
    lam_b = []
    if args.reps:
        with ProcessPoolExecutor(args.workers) as pool:
            futs = [pool.submit(_boot_rep, rec["P0"], horizons, ns, conf, args.seed, b)
                    for b in range(args.reps)]  # fmt: skip
            for k, f in enumerate(as_completed(futs), 1):
                lam_b.append(f.result()["lambda"])
                if k % 20 == 0:
                    print(f"  bootstrap {k}/{args.reps}", flush=True)
        lam_b = np.array(lam_b)
        rec["p_boot"] = float((1 + np.sum(lam_b >= rec["lambda"])) / (1 + len(lam_b)))
        rec["crit95_boot"] = float(np.quantile(lam_b, 0.95))
        rec["boot_lambda"] = lam_b
        print(f"parametric bootstrap ({args.reps} reps): p = {rec['p_boot']:.4f}, "
              f"95% critical value {rec['crit95_boot']:.3f}", flush=True)  # fmt: skip
    rec.update(csv=str(path), horizons=horizons, clones=ns, confusion=conf,
               data=_counts_summary(cohorts))  # fmt: skip
    _write(out / "result.json", rec)
    print(f"wrote {out / 'result.json'}")


# ---------------------------------------------------------------------- main
def _seeds(spec: str) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in spec.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study", choices=["null", "alt", "typing", "ratio", "boot", "data"])
    ap.add_argument("csv", nargs="?", help="data mode: CSV with columns horizon,A,B")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell names, or ratio_null / ratio_alt "
                         "(default: all cells of the study)")  # fmt: skip
    ap.add_argument("--ratios", default=",".join(f"{r:g}" for r in C.RATIOS),
                    help="ratio: assumed / true mis-typing rate ratios")  # fmt: skip
    ap.add_argument("--m", default=None, help="alt: only these m_BA values (comma-separated)")
    ap.add_argument("--seeds", default="0:10")
    ap.add_argument("--ns", default=",".join(map(str, C.NS)))
    ap.add_argument("--settings", default="default", choices=sorted(SETTINGS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--tag", default="", help="subfolder suffix, e.g. _pilot")
    ap.add_argument("--no-truth-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="data mode: seed of starts / bootstrap")
    ap.add_argument("--confusion", type=float, default=0.0, help="data mode: mis-typing rate")
    ap.add_argument("--typing-kind", default="b_as_a", help="data mode: symmetric | b_as_a")
    ap.add_argument("--out", default=None, help="data mode: output folder")
    ap.add_argument("--confusion-interval", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="data mode: the mis-typing rate is only known to lie in [LO, HI]; "
                         "test at --interval-points rates over it and report the most "
                         "conservative result (largest p-value)")  # fmt: skip
    ap.add_argument("--interval-points", type=int, default=5)
    ap.add_argument("--data-settings", default="rich", choices=sorted(SETTINGS),
                    help="data mode: fit settings")  # fmt: skip
    args = ap.parse_args()
    if args.study == "data":
        data_mode(args)
        return
    ns = tuple(int(n) for n in args.ns.split(","))
    seeds = _seeds(args.seeds)
    if args.cells in ("ratio_null", "ratio_alt"):
        names = C.study_cells(args.cells)
    elif args.cells:
        names = args.cells.split(",")
    elif args.study == "ratio":
        names = C.study_cells("ratio_null") + C.study_cells("ratio_alt")
    else:
        names = C.study_cells("null" if args.study == "boot" else args.study)
    ratios = tuple(float(r) for r in args.ratios.split(","))
    if args.m:
        keep = {float(v) for v in args.m.split(",")}
        names = [nm for nm in names if C.cell(nm)["m"] in keep]
    jobs = []
    if args.study == "boot":
        for name in names:
            for n in ns:
                for s in seeds:
                    out = args.root / f"boot{args.tag}" / name / f"n{n}_s{s}.json"
                    if out.exists() and not args.force:
                        continue
                    null = args.root / "null" / name / f"s{s}.json"
                    jobs.append((boot_task, (name, s, n, args.reps, str(out), str(null))))
    elif args.study == "ratio":
        for s in seeds:  # seed-major
            for name in names:
                out = args.root / f"typing_ratio{args.tag}" / name / f"s{s}.json"
                if ratio_done(out, ratios, ns) and not args.force:
                    continue
                jobs.append((ratio_task, (name, s, ratios, args.settings, ns, str(out),
                                          not args.no_truth_check)))  # fmt: skip
    else:
        for s in seeds:  # seed-major: early partial results cover every cell
            for name in names:
                out = args.root / f"{args.study}{args.tag}" / name / f"s{s}.json"
                if out.exists() and not args.force:
                    continue
                jobs.append((sim_task, (args.study, name, s, args.settings, ns, str(out),
                                        not args.no_truth_check)))  # fmt: skip
    print(f"{len(jobs)} tasks, {args.workers} workers", flush=True)
    t0 = time.perf_counter()
    failed = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fn, *a): a for fn, a in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
                extra = res.get("lambda", res.get("p_boot"))
                if isinstance(extra, list):
                    extra = " ".join(f"{x:.2f}" for x in extra)
                print(f"[{k}/{len(jobs)} {time.perf_counter() - t0:.0f}s] {res['seconds']:.0f}s "
                      f"{Path(res['out']).relative_to(args.root)}  {extra}", flush=True)  # fmt: skip
            except Exception as exc:  # report and continue
                failed += 1
                print(f"[{k}/{len(jobs)}] FAILED {futs[f][:3]}: {exc!r}", flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
