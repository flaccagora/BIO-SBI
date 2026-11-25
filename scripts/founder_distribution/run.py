"""Simulation check of the founder-distribution study: exact-GW NUTS and MLE at n = 512.

Data: n = 512 detected clones per generation 4/6/8/10 (XG2 design), every clone seeded by
m ~ pi (m = 1..5) cells, each A with probability p_A (numpy simulator of the
misspecification study, extended to founder sizes 1..5; true founder count and A founders
of every clone recorded). Founder distributions ``<S>_pA<p>`` as in ``fd_common``.

Models (``inference=exact_gw``, flat priors):
``est``   = founder.A and the founder-size probabilities on the true support free
            (S1: sizes {1}; S3: {1, 2, 3}; S2, S4: {1..5}), Uniform / flat Dirichlet;
``known`` = founder rows fixed at the truth (the reference: known founder distribution).
``lab<k>`` = as ``est`` with lab knowledge as an informative prior (founder-row
            Dirichlet concentrations ``1 + k * truth``, mode at the truth): founder.A ~
            Beta(1 + k p_A, 1 + k (1 - p_A)), sizes ~ Dirichlet(1 + k pi) on the true support;
            k = 20: p_A known to about +-0.1, k = 100: to about +-0.045.
``labw<k>`` = as ``lab<k>`` but the p_A prior centred 0.15 too low (confident, wrong).
MLE of the ``lab`` models = MAP: log-likelihood + log founder-row prior density in the
physical probabilities (zero for the flat ``est`` rows; transitions stay unpenalised).

NUTS as the free-founder study (adjoint kernel, 4 chains from the best of 8 L-BFGS optima,
300 warmup, tree depth <= 7, ESS/R-hat stopping rule, <= 500 draws per chain); per-clone
progenitor posterior (founder count and composition) averaged over 50 posterior draws.
MLE: multistart L-BFGS on the log-likelihood (truth start + prior draws + a high-p_A start).

    uv run python scripts/founder_distribution/run.py nuts --jobs T1_ref:S2_pA0.7:est,... \
        --workers 2 --threads 4
    uv run python scripts/founder_distribution/run.py mle --jobs ... --seeds 0,1,2,3,4,5 --workers 4

Outputs ``RESULTS/{nuts,mle}/<dist>/<model>/<gt>_s<seed>.{json,npz}`` (restartable).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RESULTS = Path("/home/matteo/BIO-SBI/results/exact_gw/founder_distribution")
N = 512
SIZE_DISTS = {
    "S1": [1.0, 0, 0, 0, 0],
    "S2": [0.7, 0.2, 0.07, 0.02, 0.01],
    "S3": [1 / 3, 1 / 3, 1 / 3, 0, 0],
    "S4": [0.2] * 5,
}
DIST_INDEX = {s: i for i, s in enumerate(SIZE_DISTS)}
SUMMARIES = ("rho", "stableA", "m_BA", "B.div")
N_PROG_DRAWS = 50
GRID = {"max_points": 3e7}


def parse_dist(d: str) -> tuple[np.ndarray, float]:
    s, pa = d.split("_pA")
    return np.asarray(SIZE_DISTS[s], float), float(pa)


def support(pi: np.ndarray) -> list[int]:
    return [m + 1 for m in range(5) if pi[m] > 0]


# ---------------------------------------------------------------------- data
def _simulate_true(P, gens, n, rng, pa, pi):
    size = rng.choice(np.arange(1, 6), size=n, p=pi)
    nA0 = rng.binomial(size, pa)
    a, b = nA0.astype(np.int64), (size - nA0).astype(np.int64)
    for _ in range(gens):
        a, b = C._step(a, b, P, rng)
        if a.max(initial=0) + b.max(initial=0) > 400_000:
            raise RuntimeError("max_cells exceeded")
    return np.stack([a, b], 1), np.stack([size, nA0], 1)


def dataset(dist: str, gt: str, seed: int):
    """Cohorts (n = 512 detected clones per generation 4/6/8/10) and true founders."""
    from bio_sbi.summaries import Cohort

    pi, pa = parse_dist(dist)
    P = C.TRUTHS[gt]
    gi = C.GTS.index(gt)
    rng = np.random.default_rng([gi, seed, 1000 + DIST_INDEX[dist.split("_")[0]],
                                 int(round(100 * pa))])  # fmt: skip
    cohorts, founders = [], []
    for h in C.HORIZONS:
        kept, kf, have, batch = [], [], 0, int(1.5 * N) + 64
        while have < N:
            tr, fo = _simulate_true(P, h, batch, rng, pa, pi)
            det = tr.sum(1) > 0
            kept.append(tr[det])
            kf.append(fo[det])
            have += int(det.sum())
            rate = max(have, 1) / (batch * len(kept))
            batch = int(min(max((N - have) / rate * 1.3, 64), 50 * N))
        cohorts.append(Cohort(float(h), np.concatenate(kept)[:N]))
        founders.append(np.concatenate(kf)[:N])
    return cohorts, founders


# ---------------------------------------------------------------------- model
def make_sim(dist: str, model: str):
    import yaml

    from bio_sbi.simulators import GaltonWatson

    pi, pa = parse_dist(dist)
    sizes = support(pi)
    conf = ROOT / "src" / "bio_sbi" / "conf" / "law" / "two_type.yaml"
    law = yaml.safe_load(conf.read_text())
    if model == "est":
        law.update(fixed={}, founder_sizes=sizes)
    elif model.startswith("lab"):
        k, pc = lab_prior(model, pa)
        law.update(fixed={}, founder_sizes=sizes, concentration={
            "founder": {"A": 1 + k * pc, "B": 1 + k * (1 - pc)},
            "founder_size": {str(m): 1 + k * float(pi[m - 1]) for m in sizes}})  # fmt: skip
    else:
        fixed = {"founder": {"A": pa, "B": 1 - pa}}
        if len(sizes) > 1:
            fixed["founder_size"] = {str(m): float(pi[m - 1]) for m in sizes}
        law.update(fixed=fixed, founder_sizes=sizes)
    return GaltonWatson(law)


def lab_prior(model: str, pa: float) -> tuple[float, float]:
    """(kappa, prior mode of p_A) of a ``lab<k>`` / ``labw<k>`` model."""
    if model.startswith("labw"):
        return float(model[4:]), max(pa - 0.15, 0.0)
    return float(model[3:]), pa


def founder_log_prior(lik, theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Log Dirichlet density of the free founder rows in the physical probabilities
    (``sum (alpha_i - 1) log pi_i``) and its gradient in ``theta`` (stick coordinates).
    """
    lp, g = 0.0, np.zeros_like(theta)
    for r, t in lik.founder_rows:
        a = np.asarray(r.alpha, float) - 1.0
        ns = r.n_sticks
        if ns == 0 or not np.any(a):
            continue
        s = theta[t : t + ns]
        tail = np.concatenate([[0.0], np.cumsum(np.log1p(-s))])
        logp = np.concatenate([np.log(s) + tail[:-1], tail[-1:]])
        lp += float(a @ logp)
        later = np.cumsum(a[::-1])[::-1][1:]  # sum_{i > j} a_i
        g[t : t + ns] = a[:-1] / s - later / (1 - s)
    return lp, g


def make_lik(dist, model, cohorts, kernel="adjoint"):
    from bio_sbi.inference import dediff_lrt as dl
    from bio_sbi.inference.exact_gw import ExactLikelihood

    sim = make_sim(dist, model)
    return sim, ExactLikelihood(sim, dl.observation(None), cohorts, grid=dict(GRID), kernel=kernel)


def truth_dict(dist: str, gt: str) -> dict:
    pi, pa = parse_dist(dist)
    t = {**C.pdict(C.TRUTHS[gt]), **C.summaries(C.TRUTHS[gt]), "founder.A": pa,
         "founder.B": 1 - pa, "mean_size": float(np.arange(1, 6) @ pi)}  # fmt: skip
    for m in support(pi):
        t[f"founder_size.{m}"] = float(pi[m - 1])
    return t


def truth_theta(sim, dist: str, gt: str) -> np.ndarray:
    from bio_sbi.inference import dediff_lrt as dl

    pi, pa = parse_dist(dist)
    P = C.TRUTHS[gt]
    m = {"A": dict(zip(dl.ENTRIES["A"], P[0], strict=True)),
         "B": dict(zip(dl.ENTRIES["B"], P[1], strict=True))}  # fmt: skip
    m["founder"] = {"A": pa, "B": 1 - pa}
    m["founder_size"] = {str(k): float(pi[k - 1]) for k in support(pi)}
    return np.clip(sim.space.encode(sim.physical_from_mapping(m)).astype(float), 1e-9, 1 - 1e-9)


def columns(sim, phys: np.ndarray) -> dict:
    names = list(sim.space.names)
    P12 = phys[:, :12].reshape(-1, 2, 6)
    summ = [C.summaries(P) for P in P12]
    cols = {n: phys[:, i] for i, n in enumerate(names)}
    for k in SUMMARIES:
        cols[k] = np.array([s[k] for s in summ])
    sz = [n for n in names if n.startswith("founder_size.")]
    if sz:
        cols["mean_size"] = sum(int(n.split(".")[1]) * cols[n] for n in sz)
    else:
        cols["mean_size"] = np.ones(len(phys))
    return cols


def stats_of(v: np.ndarray, truth: float | None) -> dict:
    mu, sd = float(v.mean()), float(v.std())
    q05, q50, q95 = (float(x) for x in np.quantile(v, [0.05, 0.5, 0.95]))
    out = {"mean": mu, "sd": sd, "q05": q05, "median": q50, "q95": q95}
    if truth is not None:
        out.update(truth=float(truth), z=(mu - truth) / sd if sd > 0 else None,
                   cover90=bool(q05 <= truth <= q95))  # fmt: skip
    return out


def _lock(out: Path) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(out.with_suffix(".lock"), os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def progenitor(lik, thetas, cohorts, founders):
    """Per-clone composition posterior averaged over ``thetas``; true founders per clone."""
    acc, labels = None, None
    for th in thetas:
        labels, probs = lik.progenitor_posterior(th)
        acc = probs if acc is None else [x + p for x, p in zip(acc, probs, strict=True)]
    avg = [x / len(thetas) for x in acc]
    per, info = [], []
    for h, c, f, pa in zip(lik.data, cohorts, founders, avg, strict=True):
        cell = {(int(a), int(b)): j for j, (a, b) in enumerate(zip(h.a, h.b, strict=True))}
        j = np.array([cell[(int(a), int(b))] for a, b in c.counts])
        per.append(pa[j])
        info.append(np.column_stack([np.full(len(j), h.generations), c.counts, f]))
    return labels, np.concatenate(per), np.concatenate(info)


# ---------------------------------------------------------------------- NUTS
def nuts_job(gt: str, dist: str, model: str, seed: int, threads: int, out_dir: Path) -> tuple:
    from bio_sbi.inference.exact_gw import ExactGW

    t0 = time.perf_counter()
    out = out_dir / "nuts" / dist / model / f"{gt}_s{seed}.json"
    if not _lock(out):
        return gt, dist, model, seed, 0.0, "locked"
    try:
        cohorts, founders = dataset(dist, gt, seed)
        sim, lik = make_lik(dist, model, cohorts)
        smoke = os.environ.get("FD_SMOKE") == "1"
        m = ExactGW(kernel="adjoint", threads=threads, chains=4, init_starts=4 if smoke else 8,
                    warmup=20 if smoke else 300, draws=400, log_every=0, seed=seed,
                    max_tree_depth=3 if smoke else 7, init_maxiter=10 if smoke else 200,
                    stop={"enabled": True, "ess": 200.0, "tail_ess": 100.0,
                          "min_draws": 20 if smoke else 100, "block": 20 if smoke else 50,
                          "max_draws": 20 if smoke else 400})  # fmt: skip
        theta, diag = m.run(lik, key=("founder_distribution", dist, model, gt, seed), stop=m.stop)
        t_nuts = time.perf_counter() - t0
        phys = sim.space.decode(theta)
        cols = columns(sim, phys)
        truth = truth_dict(dist, gt)
        res = {"gt": gt, "dist": dist, "model": model, "seed": seed, "draws": int(len(phys)),
               "stats": {k: stats_of(v, truth.get(k)) for k, v in cols.items()}}  # fmt: skip
        if model != "known":
            res["corr"] = {f"founder.A~{k}": float(np.corrcoef(cols["founder.A"], cols[k])[0, 1])
                           for k in ("B.to_A", "m_BA", "mean_size", "rho")
                           if np.std(cols[k]) > 0}  # fmt: skip
        keep = ("max_rank_rhat", "min_ess_bulk", "min_ess_tail", "max_mean_spread", "divergences",
                "warmup_divergences", "max_tree_depth_hits", "draws_per_chain", "step_size",
                "init_log_posterior", "chain_mean_log_posterior", "seconds", "seconds_per_eval",
                "gradient_evals", "chain_means")  # fmt: skip
        res["convergence"] = {k: diag.get(k) for k in keep}
        st = diag.get("stop") or {"reason": "fixed", "failed": [], "draws_per_chain": None}
        res["stop"] = {k: st[k] for k in ("reason", "failed", "draws_per_chain")}
        res["seconds_nuts"] = t_nuts
        t1 = time.perf_counter()
        rng = np.random.default_rng([seed, 11])
        idx = rng.choice(len(theta), size=min(N_PROG_DRAWS, len(theta)), replace=False)
        labels, prog, clones = progenitor(lik, theta[idx], cohorts, founders)
        _, prog_true, _ = progenitor(lik, [truth_theta(sim, dist, gt)], cohorts, founders)
        res["progenitor_labels"] = labels
        res["seconds_progenitor"] = time.perf_counter() - t1
        np.savez_compressed(
            out.with_suffix(".npz"), phys=phys.astype(np.float32), names=np.array(sim.space.names),
            chain=np.repeat(np.arange(4), len(phys) // 4).astype(np.int8),
            clones=clones.astype(np.int32),  # generation, A, B, founder count, A founders
            prog=prog.astype(np.float32), prog_true=prog_true.astype(np.float32),
            labels=np.array(labels),
        )  # fmt: skip
        res["seconds"] = time.perf_counter() - t0
        out.write_text(json.dumps(res, default=float, indent=1))
        return gt, dist, model, seed, res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".err").write_text(traceback.format_exc())
        return gt, dist, model, seed, 0.0, repr(exc)


# ---------------------------------------------------------------------- MLE
def mle_fit(lik, sim, dist: str, gt: str, rng, n_random: int = 4, maxiter: int = 400):
    from scipy.optimize import minimize

    def neg(u):
        th = 1 / (1 + np.exp(-u))
        ll, g = lik.log_likelihood(th)
        if not np.isfinite(ll):
            return 1e300, np.zeros_like(u)
        lp, gp = founder_log_prior(lik, th)
        return -(ll + lp), -(g + gp) * th * (1 - th)

    t_true = truth_theta(sim, dist, gt)
    starts = [t_true]
    for _ in range(n_random):
        starts.append(np.clip(rng.beta(lik.prior_a, lik.prior_b), 1e-4, 1 - 1e-4))
    if lik.founder_free:  # the "B founders replace dedifferentiation" branch (p_A high)
        names = list(sim.space.names)
        hi = t_true.copy()
        phys = sim.space.decode(hi)
        phys[names.index("founder.A")] = 0.95
        phys[names.index("founder.B")] = 0.05
        phys[names.index("B.to_A")] *= 0.1
        phys[names.index("B.death")] += 0.9 * sim.space.decode(t_true)[names.index("B.to_A")]
        starts.append(np.clip(sim.space.encode(phys).astype(float), 1e-6, 1 - 1e-6))
    fits = []
    for th0 in starts:
        u0 = np.clip(np.log(th0) - np.log1p(-th0), -12, 12)
        r = minimize(neg, u0, jac=True, method="L-BFGS-B", bounds=[(-14, 14)] * len(u0),
                     options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7})  # fmt: skip
        fits.append((float(-r.fun), r.x))
    fits.sort(key=lambda t: -t[0])
    return fits, float(lik.log_likelihood(t_true, grad=False)[0])


def mle_job(gt: str, dist: str, model: str, seed: int, threads: int, out_dir: Path) -> tuple:
    t0 = time.perf_counter()
    out = out_dir / "mle" / dist / model / f"{gt}_s{seed}.json"
    if not _lock(out):
        return gt, dist, model, seed, 0.0, "locked"
    try:
        cohorts, founders = dataset(dist, gt, seed)
        sim, lik = make_lik(dist, model, cohorts)
        rng = np.random.default_rng([seed, 7, C.GTS.index(gt)])
        fits, ll_true = mle_fit(lik, sim, dist, gt, rng)
        truth = truth_dict(dist, gt)
        opt = []
        for ll, u in fits:
            phys = sim.space.decode(1 / (1 + np.exp(-u)))[None]
            cols = columns(sim, phys)
            opt.append({"ll": ll, **{k: float(v[0]) for k, v in cols.items()}})
        best = opt[0]
        res = {"gt": gt, "dist": dist, "model": model, "seed": seed, "ll_truth": ll_true,
               "best": best, "optima": opt,
               "err": {k: best[k] - truth[k] for k in truth if k in best},
               "seconds": time.perf_counter() - t0}  # fmt: skip
        # progenitor posterior at the MLE (plug-in) and at the truth
        th = 1 / (1 + np.exp(-fits[0][1]))
        labels, prog, clones = progenitor(lik, [th], cohorts, founders)
        _, prog_true, _ = progenitor(lik, [truth_theta(sim, dist, gt)], cohorts, founders)
        res["progenitor_labels"] = labels
        np.savez_compressed(out.with_suffix(".npz"), clones=clones.astype(np.int32),
                            prog=prog.astype(np.float32), prog_true=prog_true.astype(np.float32),
                            labels=np.array(labels))  # fmt: skip
        out.write_text(json.dumps(res, default=float, indent=1))
        return gt, dist, model, seed, res["seconds"], None
    except Exception as exc:  # noqa: BLE001
        out.with_suffix(".err").write_text(traceback.format_exc())
        return gt, dist, model, seed, 0.0, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["nuts", "mle"])
    ap.add_argument("--jobs", required=True, help="gt:dist:model,... (model est|known)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()
    out_dir = Path(args.out)
    todo = []
    for seed in (int(s) for s in args.seeds.split(",")):
        for j in args.jobs.split(","):
            gt, dist, model = j.split(":")
            f = out_dir / args.kind / dist / model / f"{gt}_s{seed}"
            if not f.with_suffix(".json").exists() and not f.with_suffix(".lock").exists():
                todo.append((gt, dist, model, seed))
    print(f"{len(todo)} jobs", flush=True)
    fn = nuts_job if args.kind == "nuts" else mle_job
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        futs = [pool.submit(fn, *t, args.threads, out_dir) for t in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            g, d, mo, sd, sec, err = fut.result()
            print(f"[{k}/{len(todo)}] {time.time() - t0:.0f}s {d}/{mo}/{g}/s{sd} "
                  f"{'FAILED ' + err if err else f'{sec:.0f}s'}", flush=True)  # fmt: skip
    print("done", flush=True)


if __name__ == "__main__":
    main()
