"""Benchmark: exact-likelihood computation (GW, BH) vs simulation-based inference (NPE).

Every measurement is one *job*. A job runs in a fresh subprocess whose BLAS / OpenMP /
numba thread counts are fixed through environment variables before numpy is imported
(numba's thread count is also set explicitly after ``bhexact`` is imported, because that
module caps it at 2). Results are appended to ``<out>/timings.csv``.

Subcommands
-----------
``jobs [--profile full|smoke]``
    Print the job ids (one per line) of a profile.
``run JOB --out DIR [--profile P]``
    Run one job in a subprocess with the job's thread count; append rows to timings.csv.
``derive --out DIR [--mh-evals N] [--grad-evals N] [--chains C] [--datasets K]``
    Combine timings.csv (+ jobs.csv clean flags written by the shell runner) into
    ``summary.csv`` / ``summary.md``: seconds per likelihood eval, per posterior (1
    dataset), and for K datasets, per method x simulator x design x threads.

Job ids (``kind:key=value,...``)
--------------------------------
``gwlaw:n=,L=,T=``      gwexact.law (pgf iteration + FFT), one horizon, L x L grid.
``gwjac:n=,L=,T=``      profile.law_and_jac (law + 12 polynomial-extension directions).
``gwlik:design=,T=``    gwexact log-likelihood of a design (law per horizon + histogram dot).
``gwlikjac:design=,T=`` same with the 12-direction Jacobian (gradient by chain rule).
``gwprod:design=,grad=,T=`` production bio_sbi.inference.gw_exact.log_likelihood (numba,
                        data-sized damped grids, D = free coordinates); needs branch
                        exact-gw (or --gw-exact-file / $BENCH_GW_EXACT_FILE).
``bhlaw:LA=,LB=,dt=,shape=,T=``  bhexact.law_bh at [144, 240] h (one trajectory).
``bhjac:LA=,LB=,dt=,shape=,T=``  bhexact.law_bh_jac (14 tangents) at [144, 240] h.
``bhlik:LA=,LB=,dt=,shape=,T=``  bhexact.log_lik on data/real/crc0322_2d_bh.csv.
``npe:sim=,budget=,T=`` ``bio-sbi`` training run (real_like, reversible_ab, no SBC/PPC);
                        rows: simulation, training, reference sampling, total wall.
``npeinfer:sim=,budget=,T=``  amortised inference on one new dataset from the trained run
                        of the matching ``npe`` job: python+import startup, load_run,
                        summarize, and ``fitted.sample`` alone; plus the full
                        ``bio-sbi-infer --ppc 0`` CLI wall time.

Designs: real_like = GW generations {6: 165, 10: 101} / BH hours {144: 165, 240: 101};
e0_4096 = GW generations {4, 6, 10} x 4096 clones. Truth = reference rows
(gwexact.THETA_STAR; BH means A 20 h, B 30 h).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "scripts" / "b_row_theory"
REAL = {"gw": ROOT / "data/real/crc0322_2d_gw.csv", "bh": ROOT / "data/real/crc0322_2d_bh.csv"}
THREAD_VARS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"]
FIELDS = ["job", "kind", "simulator", "design", "op", "L", "dt", "shape", "threads",
          "reps", "seconds", "seconds_min", "first_call_seconds", "detail", "timestamp"]

# GW designs: generation -> clones; grid per horizon for the gwexact (undamped) law
# (support of Z_n is <= 2^n per type; n = 10 truncated at 384 as in profile.py).
GW_DESIGNS = {"real_like": {6: 165, 10: 101}, "e0_4096": {4: 4096, 6: 4096, 10: 4096}}
GW_LIK_L = {4: 32, 6: 128, 10: 384, 12: 1024}
BH_HORIZONS = [144.0, 240.0]
BH_MEANS = (20.0, 30.0)


# ---------------------------------------------------------------------- profiles
def job_list(profile: str) -> list[str]:
    """Job ids of a profile. ``smoke`` is tiny (seconds, <= 2 threads)."""
    if profile == "smoke":
        return [
            "gwlaw:n=6,L=128,T=1", "gwlaw:n=10,L=128,T=2", "gwjac:n=6,L=128,T=1",
            "gwlik:design=real_like_small,T=1", "gwlikjac:design=real_like_small,T=1",
            "gwprod:design=real_like,grad=0,T=1", "gwprod:design=real_like,grad=1,T=1",
            "bhlaw:LA=32,LB=32,dt=2,shape=4,T=1", "bhlaw:LA=32,LB=32,dt=2,shape=4,T=2",
            "bhjac:LA=16,LB=16,dt=2,shape=4,T=2", "bhlik:LA=128,LB=128,dt=2,shape=4,T=2",
            "npe:sim=gw,budget=smoke,T=2", "npeinfer:sim=gw,budget=smoke,T=2",
        ]
    threads = [1, 4, 20]
    jobs = []
    # 1. GW exact (gwexact, numpy): law per grid; Jacobian (12 directions) up to L = 1024
    #    (L = 2048 Jacobian needs ~5 GB of complex arrays: skipped).
    grids = {4: [128, 256], 6: [128, 256], 10: [384, 512, 1024, 2048], 12: [1024, 2048]}
    for T in threads:
        for n, Ls in grids.items():
            jobs += [f"gwlaw:n={n},L={L},T={T}" for L in Ls]
        for n, Ls in grids.items():
            jobs += [f"gwjac:n={n},L={L},T={T}" for L in Ls if L <= 1024]
        for d in GW_DESIGNS:
            jobs += [f"gwlik:design={d},T={T}", f"gwlikjac:design={d},T={T}"]
    # production numba kernel (single-threaded per evaluation; chains run in parallel)
    for d in ["real_like", "real_crc0322", "e0_4096"]:
        jobs += [f"gwprod:design={d},grad=0,T=1", f"gwprod:design={d},grad=1,T=1"]
    # 2. BH exact: law at [144, 240] h; Jacobian at 768x512 with 20 threads only.
    for shape in (8, 4):
        for T in threads:
            for LA, LB in [(512, 512), (768, 512), (1024, 512)]:
                for dt in (1, 2):
                    jobs.append(f"bhlaw:LA={LA},LB={LB},dt={dt},shape={shape},T={T}")
        for dt in (1, 2):
            jobs.append(f"bhjac:LA=768,LB=512,dt={dt},shape={shape},T=20")
        jobs.append(f"bhlik:LA=768,LB=512,dt=2,shape={shape},T=20")
    # 3. NPE: dev at 4 and 20 threads; full (50k) at 20 threads only (each full run is
    #    15-25 min; set BENCH_NPE_FULL_THREADS="4 20" to add 4 threads).
    full_t = [int(t) for t in os.environ.get("BENCH_NPE_FULL_THREADS", "20").split()]
    for sim in ("gw", "bh"):
        for budget, ts in (("dev", [4, 20]), ("full", full_t)):
            for T in ts:
                jobs += [f"npe:sim={sim},budget={budget},T={T}",
                         f"npeinfer:sim={sim},budget={budget},T={T}"]
    return jobs


def parse(job: str) -> tuple[str, dict]:
    kind, _, rest = job.partition(":")
    args = dict(kv.split("=", 1) for kv in rest.split(",") if kv)
    return kind, args


# ---------------------------------------------------------------------- timing
def timeit(fn, min_seconds=2.0, max_reps=50, warm=True, min_reps=3, long_call=20.0):
    """(median, min, reps, first_call) seconds of ``fn()``.

    The first call (JIT compile, caches) is timed separately and excluded when ``warm``.
    Then calls repeat until ``min_seconds`` of total time and ``min_reps`` calls (or
    ``max_reps``); a single call longer than ``long_call`` seconds ends the loop.
    """
    import statistics
    first = None
    if warm:
        t0 = time.perf_counter(); fn(); first = time.perf_counter() - t0
    ts = []
    while len(ts) < max_reps:
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        if ts[-1] > long_call or (sum(ts) >= min_seconds and len(ts) >= min_reps):
            break
    return statistics.median(ts), min(ts), len(ts), first


def row(job, kind, op, med, mn, reps, first=None, **kw):
    return {"job": job, "kind": kind, "op": op, "seconds": f"{med:.6g}",
            "seconds_min": f"{mn:.6g}", "reps": reps,
            "first_call_seconds": "" if first is None else f"{first:.6g}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **kw}


# ---------------------------------------------------------------------- GW helpers
def _gw():
    sys.path.insert(0, str(THEORY))
    import gwexact
    return gwexact


def _law_and_jac():
    """profile.law_and_jac (loaded by path: ``profile`` is also a stdlib module)."""
    import importlib.util
    _gw()
    spec = importlib.util.spec_from_file_location("b_row_profile", THEORY / "profile.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.law_and_jac


def gw_sample(P, design: dict, seed=0):
    """Conditioned-on-detection counts per generation sampled from the exact law."""
    import numpy as np
    gw = _gw()
    rng = np.random.default_rng(seed)
    out = {}
    for n, clones in design.items():
        L = max(GW_LIK_L[n], 512) if n >= 10 else GW_LIK_L[n]
        p = np.clip(gw.law(P, n, L), 0, None)
        p[0, 0] = 0
        p /= p.sum()
        idx = rng.choice(p.size, size=clones, p=p.ravel())
        out[n] = np.stack(np.unravel_index(idx, p.shape), 1)
    return out


def gw_design(name: str) -> dict:
    if name == "real_like_small":  # smoke: small grids
        return {6: 165}
    return GW_DESIGNS[name]


def gw_counts(name: str, P):
    """(generation -> (n, 2) counts) for a design (sampled at truth) or the real CSV."""
    import numpy as np
    if name == "real_crc0322":
        import pandas as pd
        t = pd.read_csv(REAL["gw"])
        return {int(h): g[["A", "B"]].to_numpy(np.int64) for h, g in t.groupby("horizon")}
    return gw_sample(P, gw_design(name))


def run_gw(job, kind, a):
    import numpy as np
    gw = _gw()
    P = gw.THETA_STAR.copy()
    T = int(a["T"])
    if kind in ("gwlaw", "gwjac"):
        n, L = int(a["n"]), int(a["L"])
        if kind == "gwlaw":
            fn = lambda: gw.law(P, n, L)  # noqa: E731
        else:
            law_and_jac = _law_and_jac()
            fn = lambda: law_and_jac(P, n, L)  # noqa: E731
        med, mn, reps, first = timeit(fn)
        op = "law" if kind == "gwlaw" else "law+jac12"
        return [row(job, kind, op, med, mn, reps, first, simulator="gw", design=f"n={n}",
                    L=L, threads=T)]
    if kind in ("gwlik", "gwlikjac"):
        design = a["design"]
        counts = gw_counts(design, P)
        hist = {n: np.unique(c, axis=0, return_counts=True) for n, c in counts.items()}
        Ls = {n: GW_LIK_L[n] for n in counts}
        if kind == "gwlik":
            def fn():
                ll = 0.0
                for n, (cells, w) in hist.items():
                    q, _ = gw.cond_logp(gw.law(P, n, Ls[n]))
                    ll += float(w @ q[cells[:, 0], cells[:, 1]])
                return ll
        else:
            law_and_jac = _law_and_jac()

            def fn():
                ll, g = 0.0, np.zeros(12)
                for n, (cells, w) in hist.items():
                    p, dp = law_and_jac(P, n, Ls[n])
                    p0 = p[0, 0]
                    pc = np.clip(p[cells[:, 0], cells[:, 1]], 1e-300, None)
                    ll += float(w @ np.log(pc)) - w.sum() * np.log1p(-p0)
                    g += dp[:, cells[:, 0], cells[:, 1]] @ (w / pc) + w.sum() * dp[:, 0, 0] / (1 - p0)
                return ll, g
        med, mn, reps, first = timeit(fn)
        detail = {"L": Ls, "clones": {n: len(c) for n, c in counts.items()}}
        return [row(job, kind, "loglik" if kind == "gwlik" else "loglik+grad12", med, mn,
                    reps, first, simulator="gw", design=design,
                    L="/".join(str(Ls[n]) for n in sorted(Ls)), threads=T,
                    detail=json.dumps(detail))]
    raise ValueError(kind)


def _gw_exact_module():
    try:
        from bio_sbi.inference import gw_exact  # noqa: F401  (branch exact-gw merged)
        return gw_exact, "bio_sbi.inference.gw_exact"
    except ImportError:
        pass
    path = os.environ.get("BENCH_GW_EXACT_FILE")
    if not path or not Path(path).exists():
        return None, None
    import importlib.util
    spec = importlib.util.spec_from_file_location("gw_exact_snapshot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gw_exact_snapshot"] = mod
    spec.loader.exec_module(mod)
    return mod, path


def run_gwprod(job, kind, a):
    import numpy as np
    gx, src = _gw_exact_module()
    if gx is None:
        return [row(job, kind, "skipped", float("nan"), float("nan"), 0, simulator="gw",
                    design=a["design"], threads=a["T"],
                    detail="gw_exact not importable: merge exact-gw or set BENCH_GW_EXACT_FILE")]
    gw = _gw()
    P12 = gw.THETA_STAR.copy()
    design = a["design"]
    counts = gw_counts(design, P12)
    two_type = design == "e0_4096"  # E0-4096 preset uses law=two_type; real_like reversible_ab
    model = gx.GWModel(
        offspring=np.array([gw.OFF[0], gw.OFF[1]], dtype=np.int64),
        n_out=np.array([6, 6], dtype=np.int64), capture=np.ones(2), confusion=np.eye(2),
        founder=np.array([1.0, 0.0]), sizes=np.array([1], dtype=np.int64),
        size_probs=np.array([1.0]), detected=True)
    P = P12.reshape(2, 6)
    dirs = [(0, j) for j in range(1, 6)] + [(1, j) for j in (range(1, 6) if two_type else (1, 2))]
    dP = np.zeros((len(dirs), 6)); rows = np.zeros(len(dirs), np.int64)
    for d, (k, j) in enumerate(dirs):
        dP[d, j], dP[d, 0], rows[d] = 1.0, -1.0, k
    data = []
    for n, c in sorted(counts.items()):
        ca, cb, w = gx.histogram(c)
        data.append(gx.HorizonData(n, ca, cb, w, gx.grid_spec(int(ca.max()), int(cb.max()))))
    grad = a["grad"] == "1"
    if not grad:
        dP, rows = dP[:0], rows[:0]
    fn = lambda: gx.log_likelihood(model, P, dP, rows, data)  # noqa: E731
    med, mn, reps, first = timeit(fn)
    grids = "/".join(f"{h.grid.Lx}x{h.grid.Ly}" for h in data)
    return [row(job, kind, f"loglik+grad{len(dirs)}" if grad else "loglik", med, mn, reps,
                first, simulator="gw", design=design, L=grids, threads=a["T"],
                detail=json.dumps({"source": src, "directions": len(dirs),
                                   "max_counts": {h.generations: [int(h.a.max()), int(h.b.max())]
                                                  for h in data}}))]


# ---------------------------------------------------------------------- BH
def run_bh(job, kind, a):
    import numpy as np
    sys.path.insert(0, str(THEORY))
    import numba
    import bhexact as bh
    import gwexact as gw
    T = int(a["T"])
    numba.set_num_threads(T)  # bhexact caps at 2 on import
    LA, LB, dt, shape = int(a["LA"]), int(a["LB"]), float(a["dt"]), int(a["shape"])
    P = gw.THETA_STAR.copy()
    # compile on a tiny grid (numba JIT, no cache), timed separately
    t0 = time.perf_counter()
    if kind == "bhjac":
        bh.law_bh_jac(P, BH_MEANS, shape, [2.0, 4.0], 8, dt=dt)
    else:
        bh.law_bh(P, BH_MEANS, shape, [2.0, 4.0], 8, dt=dt)
    compile_s = time.perf_counter() - t0
    if kind == "bhlaw":
        fn, op = (lambda: bh.law_bh(P, BH_MEANS, shape, BH_HORIZONS, (LA, LB), dt=dt)), "law"
    elif kind == "bhjac":
        fn, op = (lambda: bh.law_bh_jac(P, BH_MEANS, shape, BH_HORIZONS, (LA, LB), dt=dt)), "law+jac14"
    else:
        import pandas as pd
        t = pd.read_csv(REAL["bh"])
        counts = {float(h): g[["A", "B"]].to_numpy(np.int64) for h, g in t.groupby("horizon")}
        fn = lambda: bh.log_lik(counts, BH_HORIZONS, P, BH_MEANS, shape, (LA, LB), dt=dt)  # noqa: E731
        op = "loglik"
    med, mn, reps, _ = timeit(fn, min_seconds=1.0, max_reps=3, warm=False, min_reps=1)
    return [row(job, kind, op, med, mn, reps, compile_s, simulator="bh",
                design="real_like" if kind == "bhlik" else "h=144/240", L=f"{LA}x{LB}",
                dt=dt, shape=shape, threads=T,
                detail=json.dumps({"numba_threads": numba.get_num_threads(),
                                   "first_call_is_compile_on_tiny_grid": True}))]


# ---------------------------------------------------------------------- NPE
SIMS = {"gw": "galton_watson", "bh": "bellman_harris"}


def npe_dir(out: Path, a) -> Path:
    return out / "npe" / f"{a['sim']}_{a['budget']}_T{a['T']}"


def _bin(name):
    exe = shutil.which(name) or str(Path(sys.executable).parent / name)
    return [exe] if Path(exe).exists() else [sys.executable, "-m", "bio_sbi.cli"]


def run_npe(job, kind, a, out: Path):
    T = int(a["T"])
    run = npe_dir(out, a)
    if kind == "npe":
        if run.exists():
            shutil.rmtree(run)
        cmd = _bin("bio-sbi") + [
            f"simulator={SIMS[a['sim']]}", "law=reversible_ab", "design=real_like",
            f"budget={a['budget']}", "budget.sbc_cases=0", "budget.ppc_samples=0",
            f"torch_threads={T}", f"workers={T}", f"hydra.run.dir={run}"]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        (run.parent / f"{run.name}.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError(f"bio-sbi failed ({proc.returncode}); see {run}.log")
        m = json.loads((run / "metrics.json").read_text())
        tr = m["training"][0]
        common = dict(simulator=a["sim"], design="real_like", threads=T)
        det = json.dumps({"simulations": m["simulations"], "epochs": tr["epochs"],
                          "posterior_samples": m["posterior_samples_drawn"]})
        rows = [row(job, kind, op, s, s, 1, **common, detail=det) for op, s in [
            ("simulation", tr["simulation_seconds"]), ("training", tr["training_seconds"]),
            ("reference_sampling", m["sampling_seconds"]),
            ("pipeline_total", m["total_seconds"]), ("process_wall", wall)]]
        return rows
    # npeinfer: amortised posterior for one new dataset (the real CSV)
    if not (run / "posterior.pt").exists():
        raise RuntimeError(f"no trained run at {run}: run the matching npe job first")
    t_start = time.perf_counter()
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(T)
    from bio_sbi.pipeline import load_run, measured_cohorts
    t_import = time.perf_counter() - t_start
    t0 = time.perf_counter()
    cfg, comp, fitted = load_run(run)
    t_load = time.perf_counter() - t0
    gen = comp.generator
    t0 = time.perf_counter()
    cohorts = measured_cohorts(pd.read_csv(REAL[a["sim"]]), gen.design, list(gen.observation.labels))
    x = gen.summarize(cohorts)
    t_summ = time.perf_counter() - t0
    n = int(os.environ.get("BENCH_NPE_SAMPLES", "4096"))
    fn = lambda: fitted.sample(x, n, 0)  # noqa: E731
    med, mn, reps, first = timeit(fn, min_seconds=1.0, max_reps=10)
    common = dict(simulator=a["sim"], design="real_crc0322", threads=T)
    rows = [row(job, kind, f"sample_{n}", med, mn, reps, first, **common),
            row(job, kind, "python_import_startup", t_import, t_import, 1, **common),
            row(job, kind, "load_run", t_load, t_load, 1, **common),
            row(job, kind, "summarize", t_summ, t_summ, 1, **common)]
    # full CLI (startup + load + support mass + sampling + writing), no PPC
    cli = _bin("bio-sbi-infer")
    if cli[0].endswith("bio-sbi-infer"):
        t0 = time.perf_counter()
        proc = subprocess.run(cli + [str(run), str(REAL[a["sim"]]), "--ppc", "0",
                                     "--samples", str(n), "--out", str(run / "measured_bench")],
                              cwd=ROOT, capture_output=True, text=True)
        s = time.perf_counter() - t0
        rows.append(row(job, kind, "cli_bio-sbi-infer_ppc0", s, s, 1, **common,
                        detail=json.dumps({"exit": proc.returncode})))
    return rows


# ---------------------------------------------------------------------- driver
def execute(job: str, out: Path) -> list[dict]:
    kind, a = parse(job)
    if kind in ("gwlaw", "gwjac", "gwlik", "gwlikjac"):
        return run_gw(job, kind, a)
    if kind == "gwprod":
        return run_gwprod(job, kind, a)
    if kind in ("bhlaw", "bhjac", "bhlik"):
        return run_bh(job, kind, a)
    if kind in ("npe", "npeinfer"):
        return run_npe(job, kind, a, out)
    raise ValueError(f"unknown job kind {kind}")


def append(out: Path, rows: list[dict]):
    f = out / "timings.csv"
    new = not f.exists()
    with f.open("a", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def cmd_run(job: str, out: Path) -> int:
    """Run ``job`` in a child process with the job's thread count fixed by env vars."""
    _, a = parse(job)
    T = str(a.get("T", "1"))
    env = dict(os.environ, **{v: T for v in THREAD_VARS})
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run([sys.executable, __file__, "_exec", job, "--out", str(out)],
                          env=env).returncode


def cmd_exec(job: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = execute(job, out)
    append(out, rows)
    for r in rows:
        print(f"{job}\t{r['op']}\t{r['seconds']} s\treps={r['reps']}\t{r.get('detail', '')}")
    print(f"job wall {time.perf_counter() - t0:.1f} s")
    return 0


# ---------------------------------------------------------------------- derived table
def cmd_derive(out: Path, mh_evals: int, grad_evals: int, chains: int, datasets: int):
    """Seconds per likelihood eval, per posterior and per ``datasets`` posteriors.

    Exact MCMC: likelihood-only samplers (adaptive MH / slice) need ``mh_evals`` law
    evaluations, gradient samplers (NUTS/HMC) ``grad_evals`` law+gradient evaluations
    (defaults 4 chains x 2000 and 4 x 1000 x 10 leapfrog steps). CPU seconds = evals x
    seconds per eval. Wall: single-threaded kernels (GW) run the chains in parallel
    (wall = CPU / min(chains, cores=20)); multithreaded kernels (BH, threads = T) run the
    chains one after another (wall = CPU). NPE: training (simulation + training, once)
    + sampling per dataset (amortised).
    """
    import pandas as pd
    t = pd.read_csv(out / "timings.csv")
    t = t[t.op != "skipped"].copy()
    t["seconds"] = t["seconds"].astype(float)
    jobs = out / "jobs.csv"
    if jobs.exists():
        j = pd.read_csv(jobs)[["job", "clean"]].drop_duplicates("job", keep="last")
        t = t.merge(j, on="job", how="left")
    else:
        t["clean"] = float("nan")
    rows = []

    def add(method, sim, design, grid, threads, eval_s, eval_kind, n_evals, parallel_chains,
            clean, note=""):
        cpu = eval_s * n_evals
        wall = cpu / min(chains, 20) if parallel_chains else cpu
        rows.append({"method": method, "simulator": sim, "design": design, "grid": grid,
                     "threads": threads, "s_per_eval": eval_s, "eval": eval_kind,
                     "evals_per_posterior": n_evals, "s_per_posterior_wall": wall,
                     "s_per_posterior_cpu": cpu, f"s_{datasets}_datasets_wall": wall * datasets,
                     "clean": clean, "note": note})

    for _, r in t.iterrows():
        if r.kind in ("gwlik", "bhlaw", "bhlik") or (r.kind == "gwprod" and r.op == "loglik"):
            add(f"exact_{r.kind}_MH", r.simulator, r.design, r.L, r.threads, r.seconds,
                r.op, mh_evals, r.simulator == "gw", r.clean,
                f"dt={r['dt']},shape={r['shape']}" if r.simulator == "bh" else "")
        if r.kind in ("gwlikjac", "bhjac") or (r.kind == "gwprod" and r.op.startswith("loglik+grad")):
            add(f"exact_{r.kind}_NUTS", r.simulator, r.design, r.L, r.threads, r.seconds,
                r.op, grad_evals, r.simulator == "gw", r.clean,
                f"dt={r['dt']},shape={r['shape']}" if r.simulator == "bh" else "")
    npe = t[t.kind == "npe"]
    for (sim, threads, job), g in npe.groupby(["simulator", "threads", "job"]):
        ops = dict(zip(g.op, g.seconds))
        budget = job.split("budget=")[1].split(",")[0]
        train = ops["simulation"] + ops["training"]
        inf = t[(t.kind == "npeinfer") & (t.job == job.replace("npe:", "npeinfer:"))]
        samp = inf[inf.op.str.startswith("sample_")].seconds
        s = float(samp.iloc[0]) if len(samp) else ops["reference_sampling"]
        cli = inf[inf.op.str.startswith("cli_")].seconds
        note = (f"train once {train:.0f} s (sim {ops['simulation']:.0f} + fit "
                f"{ops['training']:.0f}); sampling {s:.3g} s/dataset"
                + (f"; CLI per dataset {float(cli.iloc[0]):.1f} s" if len(cli) else ""))
        rows.append({"method": f"npe_{budget}", "simulator": sim, "design": "real_like",
                     "grid": "", "threads": threads, "s_per_eval": float("nan"),
                     "eval": "posterior sample", "evals_per_posterior": float("nan"),
                     "s_per_posterior_wall": train + s, "s_per_posterior_cpu": float("nan"),
                     f"s_{datasets}_datasets_wall": train + datasets * s,
                     "clean": g.clean.min(), "note": note})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    with (out / "summary.md").open("w") as fh:
        fh.write(f"Exact MCMC: {mh_evals} law evals (MH/slice) or {grad_evals} law+grad evals "
                 f"(NUTS); wall assumes {chains} chains in parallel for single-threaded GW "
                 f"kernels, sequential chains for multithreaded BH.\n\n")
        fh.write(df.to_markdown(index=False, floatfmt=".4g"))
        fh.write("\n\n## Seconds per evaluation by grid and threads (median)\n\n")
        sweep = t[t.kind.isin(["gwlaw", "gwjac", "bhlaw", "bhjac"])].copy()
        if len(sweep):
            sweep["config"] = sweep.apply(
                lambda r: f"{r.kind} {r.design} L={r.L}" + (
                    f" dt={r['dt']:g} shape={r['shape']:g}" if r.simulator == "bh" else ""), axis=1)
            fh.write(sweep.pivot_table(index="config", columns="threads", values="seconds",
                                       aggfunc="median").to_markdown(floatfmt=".4g"))
            fh.write("\n")
    print((out / "summary.md").read_text())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("jobs"); s.add_argument("--profile", default="full")
    for name in ("run", "_exec"):
        s = sub.add_parser(name); s.add_argument("job"); s.add_argument("--out", required=True)
    s = sub.add_parser("derive"); s.add_argument("--out", required=True)
    s.add_argument("--mh-evals", type=int, default=4 * 2000)
    s.add_argument("--grad-evals", type=int, default=4 * 1000 * 10)
    s.add_argument("--chains", type=int, default=4)
    s.add_argument("--datasets", type=int, default=100)
    a = p.parse_args(argv)
    if a.cmd == "jobs":
        print("\n".join(job_list(a.profile)))
        return 0
    if a.cmd == "run":
        return cmd_run(a.job, Path(a.out).resolve())
    if a.cmd == "_exec":
        return cmd_exec(a.job, Path(a.out).resolve())
    return cmd_derive(Path(a.out).resolve(), a.mh_evals, a.grad_evals, a.chains, a.datasets) or 0


if __name__ == "__main__":
    sys.exit(main())
