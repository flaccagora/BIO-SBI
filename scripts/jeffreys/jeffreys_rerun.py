"""Exact-Jeffreys posterior where PSIS from Dirichlet(1/2) is unreliable (Pareto k > 0.7).

For one XG2 cell: NUTS (the Dirichlet(1/2) run's settings: 4 chains, stopping rule, warm
start from its draws) on the *surrogate* target

    L(theta) exp(s(theta))      (theta = stick coordinates, uniform reference measure),

where ``s`` is a Gaussian-process regression (squared-exponential kernel in whitened
``u = logit(theta)``, length scale 3, nugget 0.01, constant mean; bounded, so the target is
proper) of the exact ``log p_J`` at up to 2000 points where it was already evaluated (the
Dirichlet(1/2) draws and, from the second iteration, the previous rerun's draws). The
surrogate posterior is then corrected *exactly* by importance weights ``p_J / exp(s)``
(exact Jeffreys on the GPU), Pareto-smoothed. If the Pareto k of that correction is still
> 0.7, ``--iter 2`` refits the surrogate including the new draws and samples again.

Writes results/exact_gw/data_size_jeffreys/jeffreys_rerun/<gt>/n<n>_s<seed>/: draws.npz
(theta, phys, log_pj, s, w), summary.json, nuts.json (sampler diagnostics), rerun.log
(``iter{k}/`` subfolders keep earlier iterations).

  uv run python scripts/jeffreys/jeffreys_rerun.py T1_ref 128 1 [--iter 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JEFF, evaluate, jeffreys_prior  # noqa: E402

OUT = JEFF / "jeffreys_rerun"


class GPSurrogate:
    """GP regression of ``log p_J`` on whitened ``u = logit(theta)`` (value and gradient)."""

    def __init__(self, u: np.ndarray, y: np.ndarray, ell: float = 3.0, nugget: float = 0.01,
                 max_points: int = 2000, seed: int = 0) -> None:  # fmt: skip
        rng = np.random.default_rng(seed)
        if len(u) > max_points:
            pick = rng.choice(len(u), max_points, replace=False)
            u, y = u[pick], y[pick]
        self.mu = u.mean(0)
        self.W = np.linalg.inv(np.linalg.cholesky(np.cov(u.T) + 1e-9 * np.eye(u.shape[1])))
        self.x = (u - self.mu) @ self.W.T
        self.ell2 = ell**2
        self.m = float(y.mean())
        K = self._k(self.x, self.x) + nugget * np.eye(len(self.x))
        self.alpha = cho_solve(cho_factor(K), y - self.m)
        fit = self.m + self._k(self.x, self.x) @ self.alpha
        self.train_resid_sd = float(np.std(y - fit))
        self.points = len(y)

    def _k(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
        return np.exp(-0.5 * np.maximum(d2, 0) / self.ell2)

    def value(self, u: np.ndarray) -> np.ndarray:
        x = (np.atleast_2d(u) - self.mu) @ self.W.T
        return self.m + self._k(x, self.x) @ self.alpha

    def grad(self, u: np.ndarray) -> np.ndarray:
        x = (np.asarray(u) - self.mu) @ self.W.T
        k = self._k(x[None], self.x)[0] * self.alpha  # (n,)
        gx = (k[:, None] * (self.x - x)).sum(0) / self.ell2
        return self.W.T @ gx


def _mle_module():
    spec = importlib.util.spec_from_file_location(
        "exact_gw_mle", Path(__file__).resolve().parents[1] / "exact_gw_mle.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _logit(t: np.ndarray) -> np.ndarray:
    return np.log(t) - np.log1p(-t)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gt")
    ap.add_argument("n", type=int)
    ap.add_argument("seed", type=int)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iter", type=int, default=1)
    args = ap.parse_args()
    cell = f"n{args.n}_s{args.seed}"
    run = JEFF / "dir_half" / args.gt / cell
    out = OUT / args.gt / cell
    out.mkdir(parents=True, exist_ok=True)
    train_u, train_y = [], []
    z = np.load(JEFF / "jeffreys" / args.gt / cell / "jeffreys_psis.npz")
    train_u.append(_logit(z["theta"]))
    train_y.append(z["log_pj"])
    if args.iter > 1:  # keep the previous iteration, train on its points too
        prev_dir = out / f"iter{args.iter - 1}"
        prev_dir.mkdir(exist_ok=True)
        for f in ("draws.npz", "summary.json", "nuts.json", "rerun.log"):
            if (out / f).exists():
                shutil.move(str(out / f), prev_dir / f)
        zp = np.load(prev_dir / "draws.npz")
        train_u.append(_logit(zp["theta"]))
        train_y.append(zp["log_pj"])
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(name)s] %(message)s",
                        handlers=[logging.FileHandler(out / "rerun.log"),
                                  logging.StreamHandler()])  # fmt: skip
    log = logging.getLogger("jeffreys_rerun")
    t0 = time.time()

    from bio_sbi.inference.exact_gw import ExactLikelihood, load_previous_posterior
    from bio_sbi.inference.smc import psis
    from bio_sbi.pipeline import build_components

    cfg, sim, lik, names, truth, _ = _mle_module()._setup(run)
    assert str(cfg.prior) == "jeffreys", "the source run must be the Dirichlet(1/2) run"
    u0, y0 = np.concatenate(train_u), np.concatenate(train_y)
    ok = np.isfinite(y0)
    _, idx = np.unique(u0[ok], axis=0, return_index=True)
    sur = GPSurrogate(u0[ok][idx], y0[ok][idx], seed=args.iter)
    log.info("iteration %d: GP surrogate of log p_J on %d points (train residual sd %.3f)",
             args.iter, sur.points, sur.train_resid_sd)  # fmt: skip

    class SurrogateLikelihood(ExactLikelihood):
        """Likelihood x exp(s(theta)); with a uniform stick prior the NUTS target is L e^s."""

        def log_likelihood(self, theta, grad=True, weights=None):
            ll, g = super().log_likelihood(theta, grad, weights)
            th = np.asarray(theta, float)
            u = _logit(th)
            ll = ll + float(sur.value(u)[0])
            if grad:
                g = g + sur.grad(u) / (th * (1 - th))
            return ll, g

    slik = SurrogateLikelihood.__new__(SurrogateLikelihood)
    slik.__dict__.update(lik.__dict__)
    slik._lock = threading.Lock()
    slik.prior_a = np.ones_like(lik.prior_a)  # Beta(1, 1) per stick: uniform in theta
    slik.prior_b = np.ones_like(lik.prior_b)
    comp = build_components(cfg)
    method = comp.method
    method.threads = int(args.threads)
    prev = load_previous_posterior(run / "posterior.pt", sim.space)
    stop = dict(cfg.inference.method.stop)
    theta, diag = method.run(slik, key=("jeffreys",), warm=prev, stop=stop)
    theta = np.clip(theta, 1e-15, 1 - 1e-15)
    st = diag.get("stop") or {}
    log.info("NUTS: %d draws, stop %s, max rank R-hat %.3f, min ESS bulk %.0f (%.0f s)",
             len(theta), st.get("reason"), diag.get("max_rank_rhat", np.nan),
             diag.get("min_ess_bulk", np.nan), time.time() - t0)  # fmt: skip

    prior = jeffreys_prior()
    res = evaluate(prior, theta, log=None, chunk=512)
    s = sur.value(_logit(theta))
    lw = res["log_pj"] - s
    okw = np.isfinite(lw)
    lw_s, k = psis(np.where(okw, lw, -np.inf))
    w = np.exp(lw_s - lw_s[okw].max())
    w[~okw] = 0
    w /= w.sum()
    phys = sim.space.decode(theta)[:, : len(names)]
    post = {}
    for n in sim.space.free_names:
        x = phys[:, names.index(n)]
        m = float(w @ x)
        post[n] = {"mean": m, "sd": float(np.sqrt(w @ (x - m) ** 2))}
    info = {
        "run": str(run), "iteration": args.iter, "draws": int(len(theta)),
        "pareto_k": float(k), "ess": float(1 / np.sum(w**2)),
        "surrogate_points": sur.points, "surrogate_train_resid_sd": sur.train_resid_sd,
        "log_w_sd": float(np.std(lw[okw])), "seconds": time.time() - t0,
        "stop": st.get("reason"), "stop_met": st.get("met"), "stop_failed": st.get("failed"),
        "max_rank_rhat": diag.get("max_rank_rhat"), "min_ess_bulk": diag.get("min_ess_bulk"),
        "min_ess_tail": diag.get("min_ess_tail"), "divergences": diag.get("divergences"),
        "posterior": post,
    }  # fmt: skip
    np.savez_compressed(out / "draws.npz", theta=theta, phys=phys, log_pj=res["log_pj"], s=s,
                        w=w)  # fmt: skip
    (out / "summary.json").write_text(json.dumps(info, indent=1, default=str))
    keep = {k2: v for k2, v in diag.items() if isinstance(v, (int, float, str, dict, list))}
    (out / "nuts.json").write_text(json.dumps(keep, indent=1, default=str))
    log.info("exact Jeffreys correction: Pareto k %.2f, ESS %.0f / %d (%.0f s total)", k,
             info["ess"], len(theta), time.time() - t0)  # fmt: skip


if __name__ == "__main__":
    main()
