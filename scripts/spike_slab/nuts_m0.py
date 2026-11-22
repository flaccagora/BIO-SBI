"""NUTS on a face (default M0: B never divides) for a few runs: checks the M0 posterior
summaries and evidences used by the spike-and-slab analysis.

Usage:
  uv run python scripts/spike_slab/nuts_m0.py T1_ref:128:0 T1_ref:1024:0 ... [--workers 4]
      [--model M0] [--chains 4] [--warmup 500] [--draws 500]
One process per (run, chain); ALR coordinates (same target as evidence_runs.py),
starts at the face's posterior mode plus jitter. Writes
results/exact_gw/spike_slab/nuts/<gt>_n<n>_s<seed>_<model>.npz (physical draws per chain,
log targets, R-hat / ESS, divergences) and a bridge-sampling evidence from these draws.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from core import MODELS, OUT, Face, TMix, bridge, list_runs, load, weighted_gmm


def _face(key, model):
    gt, n, seed = key
    run = next(r[3] for r in list_runs() if r[:3] == (gt, n, seed))
    lik, names, truth, post, mle = load(Path(run))
    face = Face(lik, names, MODELS[model], post[0], ref_phys=post.mean(0))
    mle_phys = post.mean(0).copy()
    for k, v in mle["mle"].items():
        mle_phys[names.index(k)] = v
    return face, post, mle_phys, names, truth


def chain(key, model, c, warmup, draws):
    from bio_sbi.inference.nuts import run_chain

    t0 = time.perf_counter()
    face, post, mle_phys, names, truth = _face(key, model)
    u_hat, _ = face.mode([face.u_of(mle_phys), face.u_of(post.mean(0))])
    rng = np.random.default_rng([key[1], key[2], c, 7])
    q0 = u_hat + 0.3 * rng.standard_normal(face.dim)
    res = run_chain(face.target, q0, warmup, draws, rng, max_depth=8, dense=True)
    return {"u": res.draws, "lp": res.logp, "div": res.divergences, "evals": res.gradient_evals,
            "depth": res.mean_tree_depth, "seconds": time.perf_counter() - t0}  # fmt: skip


def finish(key, model, chains):
    from bio_sbi.inference.nuts import ess, split_rhat

    face, post, mle_phys, names, truth = _face(key, model)
    U = np.array([c["u"] for c in chains])  # (chains, draws, d)
    LP = np.array([c["lp"] for c in chains])
    phys = np.array([[face.phys(u) for u in uc] for uc in U])
    rhat = max(split_rhat(U[:, :, j]) for j in range(face.dim))
    ess_min = min(ess(U[:, :, j]) for j in range(face.dim))
    flat = U.reshape(-1, face.dim)
    lpf = LP.reshape(-1)
    rng = np.random.default_rng(11)
    out = {"phys": phys.astype(np.float32), "lp": LP, "rhat": rhat, "ess_min": ess_min,
           "divergences": sum(c["div"] for c in chains),
           "grad_evals": sum(c["evals"] for c in chains),
           "seconds": sum(c["seconds"] for c in chains), "names": np.array(names)}  # fmt: skip
    for k_mix in (1, 4, 8):
        gm = weighted_gmm(flat, np.ones(len(flat)), k_mix, rng)
        q = TMix([(c[0], c[1], c[2] * 1.21) for c in gm])
        xf = q.sample(3000, rng)
        lwf = face.targets(xf) - q.logpdf(xf)
        lwf[~np.isfinite(lwf)] = -np.inf
        out[f"logZ_bridge_k{k_mix}"] = bridge(lpf - q.logpdf(flat), lwf,
                                              n1_eff=float(ess(LP)))  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--model", default="M0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--draws", type=int, default=500)
    args = ap.parse_args()
    keys = [(g, int(n), int(s)) for g, n, s in (r.split(":") for r in args.runs)]
    dest = OUT / "nuts"
    dest.mkdir(parents=True, exist_ok=True)
    keys = [k for k in keys if not (dest / f"{k[0]}_n{k[1]}_s{k[2]}_{args.model}.npz").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {(k, c): pool.submit(chain, k, args.model, c, args.warmup, args.draws)
                for k in keys for c in range(args.chains)}  # fmt: skip
        for k in keys:
            chains = [futs[(k, c)].result() for c in range(args.chains)]
            out = finish(k, args.model, chains)
            np.savez(dest / f"{k[0]}_n{k[1]}_s{k[2]}_{args.model}.npz", **out)
            print(k, args.model, f"rhat {out['rhat']:.3f} ess {out['ess_min']:.0f} "
                  f"div {out['divergences']} {out['seconds']:.0f}s bridge "
                  + " ".join(f"{out[f'logZ_bridge_k{m}']:.2f}" for m in (1, 4, 8)), flush=True)


if __name__ == "__main__":
    main()
