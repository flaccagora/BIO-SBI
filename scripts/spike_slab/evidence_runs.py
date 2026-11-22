"""Evidence (Laplace, SMC / NUTS draws + bridge sampling) and face MLEs of every XG2 run, per model.

Usage:
  uv run python scripts/spike_slab/evidence_runs.py [--models full,M0,nodediff]
      [--workers 4] [--only T1_ref:128:0 ...] [--scale 1.0]
Writes results/exact_gw/spike_slab/runs/<gt>/n<n>_s<seed>/<model>.npz; existing files are
skipped, so the script is rerunnable (new runs with an ``mle.json`` are picked up).
One thread per worker process (adjoint kernel).
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from core import (MODELS, Face, bridge_evidence, laplace, list_runs, load, run_dir, smc_draws,
                  t_from_draws)

N_KEEP = 2000  # posterior draws stored per face model (subsampled)
NODEDIFF_SKIP = -25.0  # Laplace log BF(nodediff : full) below which no SMC is run


def do_run(gt: str, n: int, seed: int, run: str, models: list[str], scale: float) -> str:
    from bio_sbi.inference.nuts import ess

    t0 = time.perf_counter()
    out = run_dir(gt, n, seed)
    todo = [m for m in models if not (out / f"{m}.npz").exists()]
    if not todo:
        return f"{gt} n{n} s{seed}: nothing to do"
    lik, names, truth, post, mle = load(Path(run))
    out.mkdir(parents=True, exist_ok=True)
    mle_phys = post.mean(0).copy()
    for k, v in mle["mle"].items():
        mle_phys[names.index(k)] = v
    meta = {"gt": gt, "n": n, "seed": seed, "run": str(run), "names": names,
            "truth": truth.tolist(), "ll_mle_stored": mle["ll_mle"],
            "mle_stored": mle_phys.tolist(), "n_post": len(post)}  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    n_chain = 4 if len(post) % 4 == 0 else 1
    msgs = []
    for model in todo:
        t1 = time.perf_counter()
        rng = np.random.default_rng([n, seed, zlib.crc32(gt.encode()), zlib.crc32(model.encode())])
        face = Face(lik, names, MODELS[model], post[0], ref_phys=post.mean(0))
        e0 = face.evals
        starts = [face.u_of(mle_phys), face.u_of(post.mean(0))]
        starts += [face.u_of(_prior_draw(names, face.zeros, rng)) for _ in range(2)]
        save: dict = {}
        if model == "full":
            U = np.array([face.u_of(p, eps=1e-300) for p in post])
            LP = face.targets(U)
            starts.append(U[np.argmax(np.where(np.isfinite(LP), LP, -np.inf))])
            lap = laplace(face, starts)
            n1 = float(ess(LP.reshape(n_chain, -1)))
            save.update(post_lp=LP, post_ess_lp=n1)
            draws_phys = None  # the NUTS draws themselves (posterior_samples.csv.gz)
        else:  # NUTS draws projected onto the face: extra starts and reference component
            proj = np.array([face.u_of(p) for p in post[:: max(1, len(post) // 400)]])
            sub = proj[:: max(1, len(proj) // 80)]
            lp_sub = face.targets(sub)
            starts += [sub[i] for i in np.argsort(lp_sub)[::-1][:2]]
            lap = laplace(face, starts)
            lap_full = None
            if model == "nodediff" and (out / "full.npz").exists():
                lap_full = float(np.load(out / "full.npz")["logZ_laplace"])
            if lap_full is not None and lap["logZ_laplace"] - lap_full < NODEDIFF_SKIP:
                save.update(skipped_smc=True, logZ_main=lap["logZ_laplace"])
                U = LP = None
            else:
                sm = smc_draws(face, lap, rng, extra=[t_from_draws(proj, 1.0)],
                               max_evals=int(20000 * scale))  # fmt: skip
                U, LP, n1 = sm.pop("u"), sm.pop("lp"), sm.pop("ess")
                save.update(sm, post_ess_lp=n1)
            draws_phys = None
        if U is not None:
            br = bridge_evidence(face, U, LP, n1, rng, ks=(1, 4), n_fresh=int(2000 * scale))
            save.update(br, logZ_main=br["logZ_bridge"])
            if model != "full":
                keep = rng.choice(len(U), size=min(N_KEEP, len(U)), replace=False)
                draws_phys = np.array([face.phys(u) for u in U[keep]])
                save.update(post_phys=draws_phys.astype(np.float32),
                            post_logw=np.full(len(keep), -np.log(len(keep))))  # fmt: skip
        # maximum likelihood on the face
        mst = [face.phys(lap["u_hat"]), mle_phys]
        if draws_phys is not None:
            mst.append(draws_phys[0])
        f_mle, ll_mle = face.mle(mst)
        if model == "full" and mle["ll_mle"] > ll_mle:
            f_mle, ll_mle = mle_phys, mle["ll_mle"]
        lap.pop("cov")
        save.update(lap)
        save["evals"] = face.evals - e0
        save["seconds"] = time.perf_counter() - t1
        np.savez(out / f"{model}.npz", model=model, zeros=np.array(face.zeros, dtype=str),
                 mle_phys=f_mle, ll_mle=ll_mle, log_prior_phys=face.log_prior_phys, **save)
        msgs.append(f"{model}: lap {lap['logZ_laplace']:.2f} main {save['logZ_main']:.2f} "
                    f"br1 {save.get('logZ_bridge_k1', np.nan):.2f} "
                    f"br4 {save.get('logZ_bridge_k4', np.nan):.2f} "
                    f"is4 {save.get('logZ_is_k4', np.nan):.2f} "
                    f"smc {save.get('logZ_smc', np.nan):.2f} ll {ll_mle:.2f} "
                    f"{save['seconds']:.0f}s")  # fmt: skip
    return f"{gt} n{n} s{seed} ({time.perf_counter() - t0:.0f}s): " + "; ".join(msgs)


def _prior_draw(names, zeros, rng):
    p = np.zeros(len(names))
    for t in ("A", "B"):
        idx = [i for i, nm in enumerate(names) if nm.startswith(f"{t}.") and nm not in zeros]
        p[idx] = rng.dirichlet(np.ones(len(idx)))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="full,M0,nodediff")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    models = args.models.split(",")
    runs = list_runs()
    if args.only:
        want = {tuple(s.split(":")) for s in args.only}
        runs = [r for r in runs if (r[0], str(r[1]), str(r[2])) in want]
    runs = [r for r in runs if any(not (run_dir(*r[:3]) / f"{m}.npz").exists() for m in models)]
    print(f"{len(runs)} runs, models {models}, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(do_run, g, n, s, str(p), models, args.scale): (g, n, s)
                for g, n, s, p in runs}  # fmt: skip
        for k, f in enumerate(as_completed(futs), 1):
            try:
                print(f"[{k}/{len(runs)}] {f.result()}", flush=True)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                print(f"[{k}/{len(runs)}] FAILED {futs[f]}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
