"""Profile likelihoods of the 12 probabilities, B.div and m_BA for every XG2 exact-GW run.

No posterior inference is rerun: each profile point is a constrained maximum-likelihood
fit on the run's stored dataset (``reference.npz``, regenerated from the run's config and
checked bit for bit by ``scripts/exact_gw_mle.py::_setup``). Exact likelihood, adjoint
kernel, one thread per worker process.

Phases (each skips work already on disk; rerunnable):

1. ``base``: per run, log likelihood of the posterior draws of the flat (XG2), flat-B->A
   and Dirichlet(1/2) arms (starts), refit of the full-model MLE from the stored MLE and
   the 3 best draws -> ``base.npz``.
2. ``profile``: per run x functional, ``plib.run_profile`` -> ``<functional>.npz``.
3. ``global``: per run, if any profile point beats the refit MLE by > 0.01 nats, polish
   the full model from that point (new global MLE, ``global.json``) and redo every profile
   of that run from the new centre, merging with the earlier points.
4. ``polish``: per profile, sweeps / fill, then crossings alternated with a two-sided
   smoothing pass (refit each point from a higher neighbour) until stable; repairs stale
   points next to branch switches left by the crossing refinement.
5. ``cross``: every point of the other 13 profiles of a run is a feasible point for this
   profile at its own value; where one beats this profile (> 0.02 nats, drop < 5), refit
   from it, then sweep / fill / crossings again (up to ``--cross-rounds`` rounds).

Output: /home/matteo/BIO-SBI/results/exact_gw/profile_intervals/runs/<gt>/n<n>_s<seed>/.

  uv run python scripts/profile_intervals/run_profiles.py [--workers 16] [--only T1_ref:128:0 ...]
      [--funcs B.to_A ...] [--phase base|profile|global|cross|all]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from plib import FUNCTIONALS, ConstrainedModel, cross_violations, run_profile  # noqa: E402

RES = Path("/home/matteo/BIO-SBI/results/exact_gw")
XG2 = RES / "data_size"
ARMS = {"flat": XG2, "flat_btoa": RES / "data_size_flat_btoa",
        "dir_half": RES / "data_size_jeffreys" / "dir_half"}  # fmt: skip
OUT = Path(os.environ.get("PROFILE_OUT", RES / "profile_intervals"))
RUNS = OUT / "runs"
FREE = ["A.death", "A.persist", "A.to_B", "A.div_AA", "A.div_AB", "A.div_BB",
        "B.death", "B.to_A", "B.persist", "B.div_AA", "B.div_AB", "B.div_BB"]  # fmt: skip


def run_list() -> list[tuple[str, int, int]]:
    """Runs of ``runs_index.csv``; with ``PROFILE_ALL_RUNS=1`` also every other finished
    run folder ``<gt>/n<n>_s<seed>/`` (``metrics.json`` and ``mle.json`` present).
    """
    idx = pd.read_csv(XG2 / "runs_index.csv")
    keys = [(r.gt, int(r.n), int(r.seed)) for r in idx.itertuples()]
    if os.environ.get("PROFILE_ALL_RUNS") == "1":
        have = set(keys)
        for d in sorted(XG2.glob("*/n*_s*")):
            if (d / "metrics.json").exists() and (d / "mle.json").exists():
                n, s = d.name[1:].split("_s")
                k = (d.parent.name, int(n), int(s))
                if k not in have:
                    keys.append(k)
    return keys


def run_dir(key) -> Path:
    gt, n, s = key
    return XG2 / gt / f"n{n}_s{s}"


def out_dir(key) -> Path:
    gt, n, s = key
    return RUNS / gt / f"n{n}_s{s}"


_CACHE: dict = {}


def load(key):
    """``(lik, names, truth, draws, mle_json)`` for a run (cached per worker)."""
    if key in _CACHE:
        return _CACHE[key]
    from exact_gw_mle import _setup

    _cfg, _sim, lik, names, truth, _post = _setup(run_dir(key))
    lik.kernel = "adjoint"
    full = []
    for root in ARMS.values():
        f = root / key[0] / f"n{key[1]}_s{key[2]}" / "posterior_samples.csv.gz"
        if f.exists():
            full.append(pd.read_csv(f)[names].to_numpy())
    draws = np.concatenate(full)
    mle = json.loads((run_dir(key) / "mle.json").read_text())
    if len(_CACHE) > 2:
        _CACHE.clear()
    _CACHE[key] = (lik, names, truth, draws, mle)
    return _CACHE[key]


def phase_base(key) -> dict:
    f = out_dir(key) / "base.npz"
    if f.exists():
        return {"key": key, "skipped": True}
    t0 = time.perf_counter()
    lik, names, truth, draws, mle = load(key)
    e0 = lik.evals
    full = ConstrainedModel(lik, None)
    draws_ll = np.array([full.loglik(full.encode(p), grad=False)[0] for p in draws])
    stored = draws[0].copy()
    for nm, v in mle["mle"].items():
        stored[names.index(nm)] = v
    ll_stored_eval = full.loglik(full.encode(stored), grad=False)[0]
    starts = [stored] + [draws[i] for i in np.argsort(draws_ll)[::-1][:3]]
    res = [full.maximise(full.encode(s)) for s in starts]
    rb = max(res, key=lambda r: r["ll"])
    mle_phys = full.physical(rb["s"])
    out_dir(key).mkdir(parents=True, exist_ok=True)
    np.savez(f, names=np.array(names), truth=truth, draws=draws, draws_ll=draws_ll,
             mle_stored=stored, ll_stored=mle["ll_mle"], ll_stored_eval=ll_stored_eval,
             ll_truth=mle["ll_truth"], mle_refit=mle_phys, ll_refit=rb["ll"],
             refit_lls=np.array([r["ll"] for r in res]), evals=lik.evals - e0,
             seconds=time.perf_counter() - t0)  # fmt: skip
    return {"key": key, "ll_refit": rb["ll"], "ll_stored": mle["ll_mle"],
            "gain": rb["ll"] - mle["ll_mle"], "seconds": time.perf_counter() - t0}  # fmt: skip


def h0_for(func, mle: dict) -> float:
    se = (mle.get("se") or {}).get(func.name)
    if se is None or not np.isfinite(se):
        return 0.01 * func.cmax
    return float(np.clip(se / 2, 0.002, 0.05))


def save_profile(f: Path, func, r: dict, ll_ref: float, centre: np.ndarray, extra: dict):
    """Atomic write (other workers read the profiles of the same run concurrently)."""
    tmp = f.with_name(f.stem + ".tmp.npz")
    np.savez(tmp, func=func.name, cmax=func.cmax, c=r["c"], ll=r["ll"], phys=r["phys"],
             ll_ref=ll_ref, centre=centre, fits=r["fits"], evals=r["evals"],
             seconds=r["seconds"], log=np.array(r["log"], dtype=object),
             **extra)  # fmt: skip
    os.replace(tmp, f)


def phase_profile(key, fname: str) -> dict:
    f = out_dir(key) / f"{fname}.npz"
    if f.exists():
        return {"key": key, "func": fname, "skipped": True}
    func = FUNCTIONALS[fname]
    lik, names, truth, draws, mle = load(key)
    b = np.load(out_dir(key) / "base.npz")
    r = run_profile(lik, func, [b["mle_refit"]], float(b["ll_refit"]), draws,
                    b["draws_ll"], h0_for(func, mle))  # fmt: skip
    save_profile(f, func, r, float(b["ll_refit"]), b["mle_refit"], {"redone": False, "cross": 0})
    return {"key": key, "func": fname, "seconds": r["seconds"], "fits": r["fits"],
            "evals": r["evals"], "gain": float(r["ll"].max() - b["ll_refit"]),
            "points": len(r["c"])}  # fmt: skip


def best_point(key) -> tuple[float, np.ndarray, str]:
    best = (-np.inf, None, "")
    for fname in FUNCTIONALS:
        z = np.load(out_dir(key) / f"{fname}.npz", allow_pickle=True)
        j = int(np.argmax(z["ll"]))
        if z["ll"][j] > best[0]:
            best = (float(z["ll"][j]), z["phys"][j], fname)
    return best


def phase_global_check(key) -> dict:
    """Polish the full model from the best profile point; ``global.json``."""
    f = out_dir(key) / "global.json"
    if f.exists():
        return json.loads(f.read_text())
    b = np.load(out_dir(key) / "base.npz")
    ll_best, ph_best, fname = best_point(key)
    info = {"gt": key[0], "n": key[1], "seed": key[2], "ll_stored": float(b["ll_stored"]),
            "ll_refit": float(b["ll_refit"]), "ll_profile_best": ll_best,
            "profile_best_func": fname, "new_max": False}  # fmt: skip
    if ll_best > float(b["ll_refit"]) + 0.01:
        lik, names, truth, draws, mle = load(key)
        full = ConstrainedModel(lik, None)
        r = full.maximise(full.encode(ph_best))
        ph = full.physical(r["s"]) if r["ll"] >= ll_best else ph_best
        info.update(new_max=True, ll_global=max(r["ll"], ll_best),
                    mle_global={n: float(ph[i]) for i, n in enumerate(FREE)})  # fmt: skip
        np.save(out_dir(key) / "mle_global.npy", ph)
    else:
        info["ll_global"] = max(float(b["ll_refit"]), ll_best)
    f.write_text(json.dumps(info, indent=1))
    return info


def phase_redo(key, fname: str) -> dict:
    """Redo a profile from the new global MLE (merging the earlier points)."""
    f = out_dir(key) / f"{fname}.npz"
    z = np.load(f, allow_pickle=True)
    info = json.loads((out_dir(key) / "global.json").read_text())
    if bool(z["redone"]):
        return {"key": key, "func": fname, "skipped": True}
    func = FUNCTIONALS[fname]
    lik, names, truth, draws, mle = load(key)
    b = np.load(out_dir(key) / "base.npz")
    centre = np.load(out_dir(key) / "mle_global.npy")
    ll_ref = float(info["ll_global"])
    r = run_profile(lik, func, [centre], ll_ref, draws, b["draws_ll"], h0_for(func, mle),
                    init=(z["c"], z["ll"], z["phys"]))  # fmt: skip
    r["fits"] += int(z["fits"])
    r["evals"] += int(z["evals"])
    r["seconds"] += float(z["seconds"])
    save_profile(f, func, r, ll_ref, centre, {"redone": True, "cross": _cross(z)})
    return {"key": key, "func": fname, "seconds": r["seconds"], "redone": True}


def _cross(z) -> int:
    return int(z["cross"]) if "cross" in z.files else 0


def phase_cross(key, fname: str) -> dict:
    """Cross-check against the other 13 profiles of the run (``plib.cross_violations``)
    and refit this profile where another profile's point beats it.
    """
    f = out_dir(key) / f"{fname}.npz"
    z = np.load(f, allow_pickle=True)
    func = FUNCTIONALS[fname]
    curves = {g: np.load(out_dir(key) / f"{g}.npz", allow_pickle=True) for g in FUNCTIONALS}
    b = np.load(out_dir(key) / "base.npz")
    names = [str(x) for x in b["names"]]
    ll_ref = max([float(b["ll_refit"])] + [float(c["ll"].max()) for c in curves.values()])
    others = [(c["ll"], c["phys"]) for g, c in curves.items() if g != fname]
    viol = cross_violations(func, (z["c"], z["ll"]), others, names, ll_ref)
    if not viol:
        return {"key": key, "func": fname, "violations": 0}
    worst = max(v[1] - np.interp(v[0], z["c"], z["ll"]) for v in viol)
    lik, names, truth, draws, mle = load(key)
    r = run_profile(lik, func, [], ll_ref, draws, b["draws_ll"], h0_for(func, mle),
                    init=(z["c"], z["ll"], z["phys"]), points=viol)  # fmt: skip
    r["fits"] += int(z["fits"])
    r["evals"] += int(z["evals"])
    r["seconds"] += float(z["seconds"])
    save_profile(f, func, r, float(z["ll_ref"]), z["centre"],
                 {"redone": bool(z["redone"]), "cross": _cross(z) + len(viol),
                  "polished": "polished" in z.files and bool(z["polished"])})  # fmt: skip
    return {"key": key, "func": fname, "violations": len(viol), "worst": float(worst),
            "seconds": r["seconds"]}  # fmt: skip


def phase_polish(key, fname: str) -> dict:
    """Sweeps, fill, then crossings alternated with ``Profiler.smooth`` on the stored curve
    (no new outward pass), with the run's current global maximum as reference."""
    f = out_dir(key) / f"{fname}.npz"
    z = np.load(f, allow_pickle=True)
    if "polished" in z.files and bool(z["polished"]):
        return {"key": key, "func": fname, "skipped": True}
    func = FUNCTIONALS[fname]
    b = np.load(out_dir(key) / "base.npz")
    ll_ref = max([float(b["ll_refit"])] + [float(np.load(out_dir(key) / f"{g}.npz")["ll"].max())
                                          for g in FUNCTIONALS])  # fmt: skip
    lik, names, truth, draws, mle = load(key)
    before = z["ll"].copy()
    r = run_profile(lik, func, [], ll_ref, draws, b["draws_ll"], h0_for(func, mle),
                    init=(z["c"], z["ll"], z["phys"]))  # fmt: skip
    gain = float(np.max(np.interp(z["c"], r["c"], r["ll"]) - before))
    r["fits"] += int(z["fits"])
    r["evals"] += int(z["evals"])
    r["seconds"] += float(z["seconds"])
    save_profile(f, func, r, float(z["ll_ref"]), z["centre"],
                 {"redone": bool(z["redone"]), "cross": _cross(z), "polished": True})  # fmt: skip
    return {"key": key, "func": fname, "max_gain": gain, "seconds": r["seconds"]}


def _call(fn, *a):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001
        return {"args": [str(x) for x in a], "error": traceback.format_exc()}


def pool_map(tasks, workers: int, label: str):
    t0 = time.perf_counter()
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_call, *t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            r = fu.result()
            out.append(r)
            msg = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                   if k != "error"}  # fmt: skip
            print(f"[{label} {i + 1}/{len(tasks)} {time.perf_counter() - t0:.0f}s] {msg}",
                  flush=True)  # fmt: skip
            if "error" in r:
                print(r["error"], flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=None, help="gt:n:seed")
    ap.add_argument("--funcs", nargs="*", default=list(FUNCTIONALS))
    ap.add_argument("--phase", default="all", help="base|profile|global|polish|cross|all")
    ap.add_argument("--cross-rounds", type=int, default=3)
    a = ap.parse_args()
    keys = run_list()
    if a.only:
        want = {(g, int(n), int(s)) for g, n, s in (o.split(":") for o in a.only)}
        keys = [k for k in keys if k in want]
    keys.sort(key=lambda k: -k[1])  # large n first (load balance)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase in ("base", "all"):
        pool_map([(phase_base, k) for k in keys], a.workers, "base")
    if a.phase in ("profile", "all"):
        # m_BA / B.div / B.to_A are the slowest: start them first
        order = sorted(a.funcs, key=lambda f: f not in ("B.m_BA", "B.div", "B.to_A"))
        tasks = [(phase_profile, k, f) for k in keys for f in order]
        pool_map(tasks, a.workers, "profile")
    if a.phase in ("global", "all") and set(a.funcs) == set(FUNCTIONALS):
        infos = pool_map([(phase_global_check, k) for k in keys], a.workers, "global")
        redo = [(i["gt"], i["n"], i["seed"]) for i in infos if i.get("new_max")]
        print(f"new global maximum in {len(redo)} runs: {redo}", flush=True)
        tasks = [(phase_redo, k, f) for k in redo for f in FUNCTIONALS]
        if tasks:
            pool_map(tasks, a.workers, "redo")
    if a.phase in ("polish", "all"):
        res = pool_map([(phase_polish, k, f) for k in keys for f in a.funcs], a.workers,
                       "polish")  # fmt: skip
        g = [r.get("max_gain", 0) for r in res]
        print(f"polish: {sum(x > 0.01 for x in g)} profiles improved by > 0.01 nats, "
              f"max {max(g):.3f}", flush=True)  # fmt: skip
    if a.phase in ("cross", "all") and set(a.funcs) == set(FUNCTIONALS):
        for rnd in range(a.cross_rounds):
            res = pool_map([(phase_cross, k, f) for k in keys for f in FUNCTIONALS], a.workers,
                           f"cross{rnd}")  # fmt: skip
            nv = sum(r.get("violations", 0) > 0 for r in res)
            print(f"cross round {rnd}: {nv} profiles repaired", flush=True)
            if nv == 0:
                break


if __name__ == "__main__":
    main()
