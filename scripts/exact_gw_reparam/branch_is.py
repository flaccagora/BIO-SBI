"""Branch masses of an XG2 cell by importance sampling (independent check of branch_evidence).

All draws of the cell collected so far (NUTS runs, SMC populations under ``<out>/<cell>``)
are pooled, split at ``B.to_A = t`` and mapped to ``brow+probit`` coordinates; per branch a
t mixture (BIC-selected Gaussian mixture, scales x ``--scale``, ``df`` degrees of freedom)
is fitted and ``Z_b = E_q[1{v in b} exp(log post(v)) / q(v)]`` estimated from ``--n``
proposal draws (likelihood-only evaluations), with Pareto-smoothed weights and their
Pareto ``k`` (< 0.7: the estimate is trustworthy). Repeated with ``--reps`` seeds.

    uv run python scripts/exact_gw_reparam/branch_is.py --cell T1_n1024_s0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).parent))
from harness import load  # noqa: E402

from bio_sbi.inference.gw_reparam import StickCoordinates, _break  # noqa: E402
from bio_sbi.inference.smc import fit_proposal, psis  # noqa: E402

OUT = Path("/home/matteo/BIO-SBI/results/exact_gw/reparam_bench")


def collect(cell_dir: Path, names: list[str]) -> np.ndarray:
    """All physical draws stored under ``cell_dir`` (runs and SMC populations)."""
    parts = []
    for f in sorted(cell_dir.glob("*/draws.npz")):
        z = np.load(f)
        parts.append(z["physical"].reshape(-1, len(names)))
        if "smc_population_physical" in z.files:
            parts.append(z["smc_population_physical"])
    for f in sorted(cell_dir.glob("branch_evidence*/populations.npz")):
        z = np.load(f)
        parts += [z[k] for k in z.files if k != "names"]
    return np.concatenate(parts)


def to_v(coords: StickCoordinates, phys: np.ndarray) -> np.ndarray:
    """Sampler coordinates of physical vectors (rows)."""
    s = np.concatenate([_break(phys[:, cols] / mass) for _, _, _, mass, _, cols in coords.blocks],
                       axis=-1)  # fmt: skip
    return coords.from_sticks(np.clip(s, 1e-300, 1 - 1e-16))


def main() -> None:
    """Fit the proposals, importance-sample both branches, print the masses."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cell", default="T1_n1024_s0")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--n", type=int, default=16000)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--components", type=int, default=8)
    p.add_argument("--scale", type=float, default=1.3)
    p.add_argument("--df", type=float, default=4.0)
    p.add_argument("--max-fit", type=int, default=30000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = OUT / a.cell / "branch_is"
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(out / "run.log"), logging.StreamHandler()],
    )
    _, lik, _, _ = load(a.cell)
    names = list(lik.space.names)
    j = names.index("B.to_A")
    coords = StickCoordinates(lik, "brow+probit")
    kb = next(b[4].start for b in coords.blocks if b[0] == lik.sim.law.types[1])
    phys = collect(OUT / a.cell, names)
    rng = np.random.default_rng(a.seed)
    rows = []
    with ThreadPoolExecutor(a.threads) as pool:
        for branch in ("low", "high"):
            sel = phys[:, j] < a.threshold if branch == "low" else phys[:, j] >= a.threshold
            x = phys[sel]
            if len(x) > a.max_fit:
                x = x[rng.choice(len(x), a.max_fit, replace=False)]
            v = to_v(coords, x)
            v = v[np.all(np.isfinite(v), axis=1)]
            q, k_comp = fit_proposal(rng, v, np.ones(len(v)), a.components, a.scale, a.df,
                                     n_eff=len(v) / 10)  # fmt: skip
            logging.info("%s: %d pooled draws, %d-component t mixture", branch, len(v), k_comp)
            for rep in range(a.reps):
                V = q.sample(rng, a.n)
                s_to = coords.sticks(V[:, kb])[0]
                inside = s_to < a.threshold if branch == "low" else s_to >= a.threshold
                lp = np.full(a.n, -np.inf)
                idx = np.flatnonzero(inside)
                lp[idx] = np.fromiter(pool.map(coords.log_posterior_value, V[idx]), float,
                                      len(idx))  # fmt: skip
                lw = lp - q.logpdf(V)
                slw, k = psis(lw)
                log_z = float(logsumexp(slw) - np.log(a.n))
                w = np.exp(slw - np.nanmax(slw[np.isfinite(slw)]))
                w = np.where(np.isfinite(w), w, 0)
                ess = float(w.sum() ** 2 / (w**2).sum())
                rows.append({"branch": branch, "rep": rep, "log_z": log_z, "pareto_k": k,
                             "ess": ess, "inside": float(inside.mean()),
                             "components": k_comp, "fit_draws": len(v)})  # fmt: skip
                logging.info("%s rep %d: log Z %.2f, Pareto k %.2f, ESS %.0f, inside %.2f",
                             branch, rep, log_z, k, ess, inside.mean())  # fmt: skip
    lz = {b: np.array([r["log_z"] for r in rows if r["branch"] == b]) for b in ("low", "high")}
    pooled = {b: logsumexp(v) - np.log(len(v)) for b, v in lz.items()}
    mass = float(1 / (1 + np.exp(pooled["high"] - pooled["low"])))
    per = (1 / (1 + np.exp(lz["high"] - lz["low"]))).tolist()
    (out / "summary.json").write_text(json.dumps(
        {"rows": rows, "mass_low_pooled": mass, "mass_low_per_rep": per,
         "settings": {k: v for k, v in vars(a).items()}}, indent=1, default=float))  # fmt: skip
    logging.info("mass(B.to_A < %.2f): per rep %s, pooled %.4f", a.threshold, np.round(per, 4),
                 mass)  # fmt: skip


if __name__ == "__main__":
    main()
