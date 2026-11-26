"""Standard errors of the exact-GW MLE from the observed Fisher information.

For every run folder with ``mle.json`` (written by ``scripts/exact_gw_mle.py``) under the
given roots: rebuild the run's exact likelihood (dataset checked against
``reference.npz``; ``inference.method.kernel=adjoint`` for speed: identical values),
take the MLE's stick coordinates, and compute the Hessian of the log likelihood by central
differences of the analytic gradient over the coordinates strictly inside the box (a stick
at its bound, i.e. an outcome probability or a whole remainder estimated at 0, is held
fixed). The covariance ``(-H)^-1`` is mapped to the physical probabilities with the
numerical Jacobian of ``decode``; Wald standard errors are written into ``mle.json`` as
``se`` (null for probabilities whose MLE is 0: no standard error on the boundary) with
diagnostics in ``se_info`` (smallest eigenvalue of the observed information; a
non-positive one means the MLE is not a strict local maximum in some direction and the
corresponding standard errors are infinite -> stored as null).

  uv run python scripts/exact_gw_mle_se.py results/exact_gw/data_size [--workers 2]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

BOUND = 1e-6     # stick coordinate this close to 0 or 1 = on the boundary (held fixed)
H_STEP = 1e-5    # finite-difference step in stick coordinates
P_ZERO = 1e-6    # physical probability below this = estimated at 0 (no standard error)


def _mle_module():
    spec = importlib.util.spec_from_file_location(
        "exact_gw_mle", Path(__file__).with_name("exact_gw_mle.py"))  # fmt: skip
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def se_run(run: str) -> dict:
    t0 = time.perf_counter()
    run = Path(run)
    mod = _mle_module()
    from omegaconf import OmegaConf

    ov = list(OmegaConf.load(run / ".hydra" / "overrides.yaml"))
    extra = () if any(o.startswith("inference.method.kernel") for o in ov) else \
        ("inference.method.kernel=adjoint",)  # fmt: skip
    cfg, sim, lik, names, truth, post = mod._setup(run, extra)
    m = json.loads((run / "mle.json").read_text())
    free = m["parameters"]
    phys = sim.physical_from_mapping(m["mle"])
    theta = sim.encode(phys)
    d = len(theta)
    inner = [j for j in range(d) if BOUND < theta[j] < 1 - BOUND]
    fidx = [names.index(p) for p in free]

    def dphys(j):  # d physical (free entries) / d theta_j
        h = min(1e-7, 0.5 * theta[j], 0.5 * (1 - theta[j]))
        e = np.zeros(d)
        e[j] = h
        return (sim.decode(theta + e)[fidx] - sim.decode(theta - e)[fidx]) / (2 * h)

    # sticks after a stick at 1 have no effect on any probability (their outcomes are 0):
    # exactly flat, irrelevant directions -> drop them
    cols = {j: dphys(j) for j in inner}
    inner = [j for j in inner if np.max(np.abs(cols[j])) > 1e-10]

    def grad(th):
        return lik.log_likelihood(np.clip(th, 1e-12, 1 - 1e-12))[1]

    H = np.zeros((len(inner), len(inner)))
    for a, j in enumerate(inner):
        h = min(H_STEP, 0.5 * theta[j], 0.5 * (1 - theta[j]))
        e = np.zeros(d)
        e[j] = h
        H[:, a] = (grad(theta + e)[inner] - grad(theta - e)[inner]) / (2 * h)
    H = 0.5 * (H + H.T)
    info = -H
    w, V = np.linalg.eigh(info) if len(inner) else (np.zeros(0), np.zeros((0, 0)))
    ok = w > 1e-10 * max(1.0, float(np.max(np.abs(w))) if len(w) else 1.0)
    # covariance on the well-determined subspace; directions with non-positive curvature
    # give infinite variance for every parameter that loads on them
    cov = (V[:, ok] / w[ok]) @ V[:, ok].T if len(inner) else np.zeros((0, 0))
    bad_dirs = V[:, ~ok] if len(inner) else np.zeros((0, 0))
    J = np.column_stack([cols[j] for j in inner]) if inner else np.zeros((len(fidx), 0))
    var = np.einsum("ia,ab,ib->i", J, cov, J) if len(inner) else np.zeros(len(fidx))
    # a parameter has infinite standard error if it moves along a direction of zero or
    # negative curvature (relative loading > 1e-6 of its largest Jacobian entry)
    if bad_dirs.size:
        load = np.abs(J @ bad_dirs)
        scale = np.maximum(np.abs(J).max(axis=1, keepdims=True), 1e-300)
        infinite = (load / scale > 1e-6).any(axis=1)
    else:
        infinite = np.zeros(len(fidx), bool)
    se = {}
    for k, p in enumerate(free):
        if phys[fidx[k]] < P_ZERO or infinite[k] or var[k] < 0:
            se[p] = None
        else:
            se[p] = float(np.sqrt(var[k]))
    m["se"] = se
    m["se_info"] = {
        "method": "observed Fisher information (central differences of the analytic gradient)",
        "inner_coordinates": len(inner), "coordinates": d,
        "min_eigenvalue": float(w.min()) if len(w) else None,
        "non_positive_directions": int((~ok).sum()) if len(w) else 0,
        "at_zero": [p for k, p in enumerate(free) if phys[fidx[k]] < P_ZERO],
        "seconds": time.perf_counter() - t0,
    }  # fmt: skip
    (run / "mle.json").write_text(json.dumps(m, indent=2))
    return {"run": str(run), "nan": sum(v is None for v in se.values()),
            "neg": m["se_info"]["non_positive_directions"], "seconds": m["se_info"]["seconds"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    runs = [str(p.parent) for r in args.roots for p in sorted(r.rglob("mle.json"))
            if (p.parent / ".hydra" / "overrides.yaml").exists()]  # fmt: skip
    print(f"{len(runs)} runs", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(se_run, r): r for r in runs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                res = f.result()
                print(f"[{k}/{len(runs)}] {res['seconds']:.0f}s no-se={res['nan']} "
                      f"flat-dirs={res['neg']} {res['run']}", flush=True)  # fmt: skip
            except Exception as exc:
                print(f"[{k}/{len(runs)}] FAILED {futs[f]}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
