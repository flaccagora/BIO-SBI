"""PSIS reweighting of the Dirichlet(1/2) XG2 posteriors to the exact Jeffreys prior.

For every finished cell ``<dir_half>/<gt>/n<n>_s<seed>`` (NUTS under ``prior=jeffreys``,
i.e. a symmetric Dirichlet(1/2) on both rows): all draws of ``posterior.pt`` (stick
coordinates theta, chain-major), ``log p_J(theta)`` (exact Jeffreys, unnormalised; GPU) and
the Dirichlet(1/2) log density in theta; log weights ``log p_J - log p_Dir(1/2)``,
Pareto-smoothed. Writes ``<out>/<gt>/n<n>_s<seed>/jeffreys_psis.npz`` (theta, chain,
log_pj, log_dir, diagnostics, smoothed normalised weights) and ``summary.json`` (Pareto
k, ESS, weighted posterior mean / sd / 5-95% of every free physical probability).

  uv run python scripts/jeffreys/jeffreys_psis.py [--watch]   # --watch: poll for new cells
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JEFF, evaluate, jeffreys_prior  # noqa: E402

SRC = JEFF / "dir_half"
OUT = JEFF / "jeffreys"


def weighted_summary(x: np.ndarray, w: np.ndarray) -> dict:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    q = {f"q{int(p * 100):02d}": float(x[order][np.searchsorted(cw, p * cw[-1])])
         for p in (0.05, 0.5, 0.95)}  # fmt: skip
    m = float(np.sum(w * x))
    return {"mean": m, "sd": float(np.sqrt(np.sum(w * (x - m) ** 2))), **q}


def process(prior, run: Path, out: Path) -> dict:
    from bio_sbi.inference.exact_gw import load_previous_posterior
    from bio_sbi.inference.smc import psis

    sim = prior.lik.sim
    prev = load_previous_posterior(run / "posterior.pt", sim.space)
    theta = np.clip(prev.theta, 1e-15, 1 - 1e-15)
    t0 = time.time()
    res = evaluate(prior, theta, log=None, chunk=512)
    secs = time.time() - t0
    log_dir = np.array([prior.log_dirichlet(t, 0.5) for t in theta])
    lw = res["log_pj"] - log_dir
    ok = np.isfinite(lw)
    lw_s, k = psis(np.where(ok, lw, -np.inf))
    w = np.exp(lw_s - lw_s[ok].max())
    w[~ok] = 0.0
    w /= w.sum()
    names = list(sim.parameter_names)
    phys = np.array([sim.decode(t) for t in theta])
    free = [n for n in sim.space.free_names]
    summ = {n: weighted_summary(phys[:, names.index(n)], w) for n in free}
    raw = np.exp(lw[ok] - lw[ok].max())
    info = {
        "run": str(run), "draws": int(len(theta)), "finite": int(ok.sum()),
        "pareto_k": float(k), "ess": float(1 / np.sum(w**2)),
        "ess_raw": float(raw.sum() ** 2 / np.sum(raw**2)),
        "log_w_sd": float(np.std(lw[ok])), "seconds": secs,
        "min_eig_min": float(res["min_eig"].min()), "omitted_max": float(res["omitted"].max()),
        "grid10_max": int(res["grid10"].max()), "posterior": summ,
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "jeffreys_psis.npz", theta=theta,
                        chain=prev.chain if prev.chain is not None else np.zeros(0),
                        log_pj=res["log_pj"], log_dir=log_dir, w=w, min_eig=res["min_eig"],
                        omitted=res["omitted"], grid10=res["grid10"])  # fmt: skip
    (out / "summary.json").write_text(json.dumps(info, indent=1))
    return info


def pending() -> list[Path]:
    runs = []
    for m in sorted(SRC.glob("*/n*_s*/metrics.json")):
        run = m.parent
        if not (run / "posterior.pt").exists():
            continue
        if (OUT / run.parent.name / run.name / "summary.json").exists():
            continue
        runs.append(run)
    # curated truths and small n first (same order as the jobs file)
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--watch", action="store_true", help="poll for new cells until all 108")
    ap.add_argument("--total", type=int, default=108)
    args = ap.parse_args()
    prior = jeffreys_prior()
    while True:
        for run in pending():
            info = process(prior, run, OUT / run.parent.name / run.name)
            print(
                f"{time.strftime('%F %T')} {run.parent.name}/{run.name}: k {info['pareto_k']:.2f} "
                f"ESS {info['ess']:.0f}/{info['draws']} ({info['seconds']:.0f}s)",
                flush=True,
            )
        done = len(list(OUT.glob("*/n*_s*/summary.json")))
        if not args.watch or done >= args.total:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
