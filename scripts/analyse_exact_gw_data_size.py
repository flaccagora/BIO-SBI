"""XG2 analysis: exact GW posterior vs data size, ground truth and prior.

Reads every run under the given sweep roots (preset ``exact_gw_data_size``; reruns with
``prior=<name>`` may live in further roots) and writes, into the first root:

* ``xg2_runs.csv`` - one row per (gt, seed, n, target prior, parameter): posterior
  mean / sd / 90% interval, covered, z, contraction under the target prior, shift from
  the flat posterior in flat-posterior sd, PSIS Pareto k and ESS of the reweighting
  (``source`` = ``mcmc`` for the run's own prior, ``psis`` for reweighted, ``rerun``
  when a run with that prior exists), plus the run's R-hat / divergences.
* ``xg2_nstar.csv`` - per (gt, seed, parameter): the smallest n from which on (all
  larger n too) the convergence criteria hold; ``inf`` = not reached by the largest n.
* ``xg2_bvm.csv`` - posterior sd x sqrt(n) per (gt, parameter, n), mean over seeds
  (flat): levels off at the Cramer-Rao value once the posterior is in the
  Bernstein-von Mises regime.
* ``xg2_info.csv`` - information gain per run and row (A, B), Gaussian approximation
  sum_i 0.5 log(prior var_i / posterior var_i) in nats (marginal; ignores correlations).
* ``xg2_coverage.csv`` - 90% coverage per (gt, n, prior) over seeds and parameters.
* ``xg2_reruns.txt`` - commands for (run, prior) pairs with Pareto k > 0.7.

Criteria (per parameter, run; options below): sampler ok (max R-hat <= 1.05),
contraction >= 0.9 under the flat prior, and prior insensitivity: shift <= 0.25 flat sd
for every *symmetric* prior (sparse, jeffreys, smooth, central) that is reliable (k <= 0.7
or rerun). The concentrated ``wrong`` prior gets its own n* (``nstar_wrong``).

Usage:
  uv run python scripts/analyse_exact_gw_data_size.py results/exact_gw/data_size \
      [results/exact_gw/data_size_reruns] [--contraction 0.9] [--shift 0.25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

SYMMETRIC = ("sparse", "jeffreys", "smooth", "central")
K_MAX = 0.7


# ------------------------------------------------------------------------------ PSIS
def _gpdfit(x: np.ndarray) -> tuple[float, float]:
    """Generalised Pareto fit (Zhang & Stephens 2009, with the loo weak prior on k)."""
    n = len(x)
    m = 30 + int(np.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= 3 * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b / k)) - k - 1)
    w = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep] / w[keep].sum(), b[keep]
    b_post = float(np.sum(b * w))
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    k_post = (n * k_post + 10 * 0.5) / (n + 10)
    return k_post, sigma


def _gpinv(p: np.ndarray, k: float, sigma: float) -> np.ndarray:
    if abs(k) < 1e-12:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(log_w: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Pareto-smoothed normalised weights, Pareto k and ESS (Vehtari et al. 2024)."""
    lw = np.asarray(log_w, float) - np.max(log_w)
    s = len(lw)
    m = int(min(0.2 * s, 3 * np.sqrt(s)))
    order = np.argsort(lw)
    k = 0.0
    if m >= 5:
        tail_idx, cut = order[-m:], lw[order[-m - 1]]
        tail = np.exp(lw[tail_idx]) - np.exp(cut)
        if np.ptp(tail) > 0 and tail[-1] > 0:
            k, sigma = _gpdfit(np.maximum(tail, 1e-300))
            if np.isfinite(k):
                smooth = _gpinv((np.arange(1, m + 1) - 0.5) / m, k, sigma) + np.exp(cut)
                lw[tail_idx] = np.minimum(np.log(np.maximum(smooth, 1e-300)), 0.0)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    return w, float(k), float(1 / np.sum(w**2))


# ---------------------------------------------------------------------- run loading
def _alpha(conc: dict, name: str) -> float:
    row, entry = name.split(".", 1)
    return float((conc.get(row) or {}).get(entry, 1.0))


def _prior_var(conc: dict, names: list[str]) -> dict[str, float]:
    """Marginal variance of each free entry under its row's Dirichlet."""
    out = {}
    for n in names:
        row = n.split(".")[0]
        a0 = sum(_alpha(conc, m) for m in names if m.split(".")[0] == row)
        a = _alpha(conc, n)
        out[n] = a * (a0 - a) / (a0**2 * (a0 + 1))
    return out


def load_run(path: Path) -> dict | None:
    cfg_path = path / ".hydra" / "config.yaml"
    if not (path / "metrics.json").exists() or not cfg_path.exists():
        return None
    cfg = OmegaConf.load(cfg_path)
    metrics = json.loads((path / "metrics.json").read_text())
    rec = pd.read_csv(path / "recovery.csv")
    free = rec.loc[rec["inferred"], "parameter"].tolist()
    draws = pd.read_csv(path / "posterior_samples.csv.gz")[free]
    priors = OmegaConf.to_container(cfg.priors)
    ex = metrics.get("exact_gw") or {}
    return {
        # dataset index: data_seed when set (reruns with another sampler seed), else seed
        "path": str(path), "gt": str(cfg.gt),
        "seed": int(cfg.data_seed if cfg.get("data_seed") is not None else cfg.seed),
        "n": int(cfg.design.clones), "prior": str(cfg.prior), "priors": priors,
        "free": free, "truth": dict(zip(rec["parameter"], rec["truth"])), "draws": draws,
        "rhat": ex.get("max_rhat", np.nan), "divergences": ex.get("divergences", np.nan),
        "minutes": metrics.get("total_seconds", np.nan) / 60,
    }  # fmt: skip


def summarise(run: dict, target: str, source: str) -> tuple[list[dict], float, float]:
    """Posterior summaries of ``run`` under prior ``target`` (reweighting if needed)."""
    free, draws, priors = run["free"], run["draws"], run["priors"]
    base, new = priors[run["prior"]], priors[target]
    x = draws.to_numpy()
    if target == run["prior"]:
        w, k, ess = np.full(len(x), 1 / len(x)), 0.0, float(len(x))
    else:
        log_w = sum((_alpha(new, n) - _alpha(base, n)) * np.log(x[:, i])
                    for i, n in enumerate(free))  # fmt: skip
        w, k, ess = psis(log_w)
    pvar = _prior_var(new, free)
    rows = []
    for i, n in enumerate(free):
        v = x[:, i]
        mean = float(w @ v)
        sd = float(np.sqrt(max(w @ (v - mean) ** 2, 0.0)))
        order = np.argsort(v)
        cdf = np.cumsum(w[order])
        q05, q95 = (float(v[order][np.searchsorted(cdf, q)]) for q in (0.05, 0.95))
        truth = float(run["truth"][n])
        rows.append({
            "gt": run["gt"], "seed": run["seed"], "n": run["n"], "prior": target,
            "source": source, "parameter": n, "row": n.split(".")[0], "truth": truth,
            "mean": mean, "sd": sd, "q05": q05, "q95": q95,
            "covered": bool(q05 <= truth <= q95), "z": (mean - truth) / sd if sd else np.nan,
            "contraction": 1 - sd**2 / pvar[n], "prior_var": pvar[n],
            "pareto_k": k, "ess": ess, "rhat": run["rhat"],
            "divergences": run["divergences"], "minutes": run["minutes"],
        })  # fmt: skip
    return rows, k, ess


def first_stable(ns: list[int], ok: list[bool]) -> float:
    """Smallest n such that the criterion holds at n and every larger n."""
    best = np.inf
    for n, good in sorted(zip(ns, ok), reverse=True):
        if not good:
            break
        best = n
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--contraction", type=float, default=0.9)
    ap.add_argument("--shift", type=float, default=0.25)
    ap.add_argument("--rhat", type=float, default=1.05)
    args = ap.parse_args()

    runs = [r for root in args.roots for p in sorted(root.rglob("metrics.json"))
            if (r := load_run(p.parent)) is not None]  # fmt: skip
    if not runs:
        raise SystemExit("no finished runs found")
    key = lambda r: (r["gt"], r["seed"], r["n"])  # noqa: E731
    by_prior = {(key(r), r["prior"]): r for r in runs}
    flat_runs = [r for r in runs if r["prior"] == "flat"]
    rows, reruns = [], []
    for r in flat_runs:
        for target in r["priors"]:
            rerun = by_prior.get((key(r), target))
            if target != "flat" and rerun is not None:
                out, _, _ = summarise(rerun, target, "rerun")
            else:
                out, k, _ = summarise(r, target, "mcmc" if target == "flat" else "psis")
                if k > K_MAX:
                    reruns.append(
                        f"bio-sbi +experiment=exact_gw_data_size prior={target} gt={r['gt']} "
                        f"seed={r['seed']} design.clones={r['n']} "
                        f"hydra.run.dir=results/exact_gw/data_size_reruns/"
                        f"{r['gt']}_n{r['n']}_s{r['seed']}_{target}  # k = {k:.2f}"
                    )
            rows.extend(out)
    df = pd.DataFrame(rows)
    flat = df[df.prior == "flat"].set_index(["gt", "seed", "n", "parameter"])
    df = df.join(flat[["mean", "sd"]].rename(columns={"mean": "flat_mean", "sd": "flat_sd"}),
                 on=["gt", "seed", "n", "parameter"])  # fmt: skip
    df["shift_sd"] = (df["mean"] - df["flat_mean"]).abs() / df["flat_sd"]
    df["reliable"] = (df.source != "psis") | (df.pareto_k <= K_MAX)
    out = args.roots[0]
    df.to_csv(out / "xg2_runs.csv", index=False)

    # ---- criteria and n*
    sym = df[df.prior.isin(SYMMETRIC)]
    sym_shift = sym.groupby(["gt", "seed", "n", "parameter"]).apply(
        lambda g: g.shift_sd.max() if g.reliable.all() else np.nan, include_groups=False)
    wrong = df[df.prior == "wrong"].set_index(["gt", "seed", "n", "parameter"])
    crit = flat[["truth", "mean", "sd", "contraction", "covered", "rhat"]].copy()
    crit["sym_shift"] = sym_shift
    crit["wrong_shift"] = wrong["shift_sd"].where(wrong["reliable"])
    crit["ok_sampler"] = crit.rhat <= args.rhat
    crit["ok_contraction"] = crit.contraction >= args.contraction
    crit["ok_prior"] = crit.sym_shift <= args.shift  # NaN (unreliable) -> False
    crit["ok"] = crit.ok_sampler & crit.ok_contraction & crit.ok_prior
    crit["ok_wrong"] = crit.ok & (crit.wrong_shift <= args.shift)
    crit = crit.reset_index()
    nstar = []
    for (gt, seed, par), g in crit.groupby(["gt", "seed", "parameter"]):
        ns = g.n.tolist()
        nstar.append({
            "gt": gt, "seed": seed, "parameter": par,
            "nstar": first_stable(ns, g.ok.tolist()),
            "nstar_contraction": first_stable(ns, g.ok_contraction.tolist()),
            "nstar_prior": first_stable(ns, g.ok_prior.tolist()),
            "nstar_wrong": first_stable(ns, g.ok_wrong.tolist()),
        })  # fmt: skip
    nstar = pd.DataFrame(nstar)
    nstar.to_csv(out / "xg2_nstar.csv", index=False)

    # ---- BvM scaling, information gain, coverage
    fl = df[df.prior == "flat"].copy()
    fl["sd_sqrt_n"] = fl.sd * np.sqrt(fl.n)
    bvm = fl.groupby(["gt", "parameter", "n"])["sd_sqrt_n"].mean().unstack("n")
    bvm.to_csv(out / "xg2_bvm.csv")
    fl["info_nats"] = 0.5 * np.log(fl.prior_var / fl.sd**2)
    info = fl.groupby(["gt", "seed", "n", "row"])["info_nats"].sum().unstack("row")
    info.to_csv(out / "xg2_info.csv")
    cov = df[df.reliable].groupby(["gt", "n", "prior"])["covered"].mean().unstack("prior")
    cov.to_csv(out / "xg2_coverage.csv")
    (out / "xg2_reruns.txt").write_text("\n".join(reruns) + ("\n" if reruns else ""))

    # ---- console summary
    pd.set_option("display.width", 200)
    print(f"{len(flat_runs)} flat runs, {len(runs) - len(flat_runs)} reruns; "
          f"{len(reruns)} (run, prior) pairs need a rerun (k > {K_MAX})")  # fmt: skip
    med = nstar.groupby(["parameter", "gt"])["nstar"].median().unstack("gt")
    print("\nn* (median over seeds; inf = not converged by the largest n):\n", med)
    print("\nflat contraction, worst parameter (mean over seeds):\n",
          fl.groupby(["gt", "seed", "n"])["contraction"].min().groupby(["gt", "n"]).mean().unstack("n").round(2))
    print("\ninformation gain, nats (mean over seeds):\n",
          info.groupby(["gt", "n"]).mean().round(1))
    mle_summary(flat_runs, out)


def mle_summary(flat_runs: list[dict], out: Path) -> None:
    """MLE (scripts/exact_gw_mle.py, ``mle.json``) vs posterior mean, if available."""
    rows = []
    for r in flat_runs:
        path = Path(r["path"]) / "mle.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        for p in m["parameters"]:
            t = m["truth"][p]
            rows.append({
                "gt": r["gt"], "seed": r["seed"], "n": r["n"], "parameter": p, "truth": t,
                "mle": m["mle"][p], "posterior_mean": m["posterior_mean"][p],
                "abs_err_mle": abs(m["mle"][p] - t),
                "abs_err_post": abs(m["posterior_mean"][p] - t),
                "ll_gap_truth": m["ll_mle"] - m["ll_truth"],
                "ll_gap_post_mean": m["ll_mle"] - m["ll_posterior_mean"],
                "missed_optimum": m["missed_optimum"],
                "starts_at_best": m["starts_within_0.01_of_best"],
            })  # fmt: skip
    if not rows:
        return
    mle = pd.DataFrame(rows)
    mle.to_csv(out / "xg2_mle.csv", index=False)
    print(f"\n== MLE vs posterior mean ({mle.groupby(['gt', 'seed', 'n']).ngroups} runs with "
          "mle.json; mean |error| over seeds)")  # fmt: skip
    b = mle[mle.parameter == "B.to_A"]
    tab = b.groupby(["gt", "n"])[["abs_err_mle", "abs_err_post"]].mean().unstack("n").round(3)
    print("B.to_A:\n", tab)
    allp = mle.groupby(["gt", "n"])[["abs_err_mle", "abs_err_post"]].mean().unstack("n").round(3)
    print("all free parameters:\n", allp)
    runs = mle.groupby(["gt", "seed", "n"]).first()
    print("log-likelihood gap MLE - truth (nats; 95% LR region for 10 free ~ 9.2), median:\n",
          runs.groupby(["gt", "n"])["ll_gap_truth"].median().unstack("n").round(1))
    print(f"runs where an optimisation started at the truth beat the multistart MLE: "
          f"{int(runs.missed_optimum.sum())}")  # fmt: skip


if __name__ == "__main__":
    main()
