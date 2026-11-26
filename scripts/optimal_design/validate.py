"""Checks of the information-based design predictions.

1. Posteriors (current design, A-founded 4/6/8/10): XG2 NUTS posterior sd of B->A vs the
   Gaussian prediction from the per-clone information (flat prior and asymptotic), all XG2
   runs (n per endpoint 256..4096; posterior sds from the weak-directions study's table).
2. MLE simulation: datasets simulated from the GW model (B-founded clones included; own
   numpy simulator, conditioned on detection), exact-likelihood MLE (L-BFGS-B in the stick
   coordinates, started at the truth: a local, asymptotic check), empirical sd of the MLE of
   B->A and of its projection on the weakest eigen-direction vs the prediction.

  uv run --with cvxpy python scripts/optimal_design/validate.py [posterior|mle|all] [n_datasets]
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from od_common import (  # noqa: E402
    BOUNDARY, CACHE, I_BTOA, NAMES, OFF, OUT, POINTS, PRIOR_PREC, REPO, SHORTNAME, TAB,
    TRUTHS, c_vector, design_truth,
)  # fmt: skip

LAB = [f"{f}{h}" for f, h in POINTS]
WD_TAB = REPO / "docs" / "weak_directions" / "tables"


def load(label):
    z = np.load(CACHE / f"{label}.npz")
    return z["K"].reshape(-1, 10, 10), c_vector(z["U"])


def design_K(Ks, w):
    return np.tensordot(w, Ks, axes=1)


def current_w():
    w = np.zeros(len(POINTS))
    for h in (4, 6, 8, 10):
        w[LAB.index(f"A{h}")] = 0.25
    return w


# ------------------------------------------------------------------ 1. posterior check
def posterior_check() -> pd.DataFrame:
    pa = pd.read_csv(WD_TAB / "posterior_alignment.csv")
    rows = []
    for gt in TRUTHS:
        Ks, c = load(gt)
        K = design_K(Ks, current_w())
        sub = pa[pa.truth == gt]
        for _, r in sub.iterrows():
            N = 4 * int(r.n)
            v_prior = c @ np.linalg.solve(np.eye(10) + N * K, c) / PRIOR_PREC
            v_asym = c @ np.linalg.solve(N * K, c) / PRIOR_PREC
            rows.append({"truth": SHORTNAME[gt], "n per endpoint": int(r.n), "seed": int(r.seed),
                         "sd post": r.sd_BtoA_post, "sd pred (prior)": np.sqrt(v_prior),
                         "sd pred (asymptotic)": np.sqrt(v_asym),
                         "sd pred weak-dir study": r.sd_BtoA_gauss_truth,
                         "var ratio post/pred(prior)": r.sd_BtoA_post**2 / v_prior})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "validation_posterior.csv", index=False)
    g = df.groupby(["truth", "n per endpoint"], sort=False).agg(
        sd_post=("sd post", "mean"), sd_pred_prior=("sd pred (prior)", "first"),
        sd_pred_asym=("sd pred (asymptotic)", "first"),
        var_ratio=("var ratio post/pred(prior)", "median")).reset_index()  # fmt: skip
    with open(TAB / "validation_posterior.md", "w") as f:
        f.write(g.to_markdown(index=False, floatfmt=".3g"))
        f.write(f"\n\nAll runs: median variance ratio {df['var ratio post/pred(prior)'].median():.2f}, "
                f"IQR {df['var ratio post/pred(prior)'].quantile(.25):.2f}-"
                f"{df['var ratio post/pred(prior)'].quantile(.75):.2f}; "
                f"n per endpoint >= 1024: median "
                f"{df[df['n per endpoint'] >= 1024]['var ratio post/pred(prior)'].median():.2f}\n")  # fmt: skip
    return df


# ------------------------------------------------------------------ 2. MLE simulation
def simulate(p12, founder, h, n, rng) -> np.ndarray:
    """``n`` detected clones (counts (A, B)) founded by one ``founder`` cell, generation h."""
    P = np.asarray(p12, float).reshape(2, 6)
    out = []
    need = n
    while need > 0:
        m = int(need * 1.5) + 16
        a = np.full(m, 1 if founder == "A" else 0, np.int64)
        b = np.full(m, 0 if founder == "A" else 1, np.int64)
        for _ in range(h):
            oa = rng.multinomial(a, P[0])
            ob = rng.multinomial(b, P[1])
            a = oa @ OFF[:, 0] + ob @ OFF[:, 0]
            b = oa @ OFF[:, 1] + ob @ OFF[:, 1]
        keep = (a + b) > 0
        got = np.stack([a[keep], b[keep]], 1)[:need]
        out.append(got)
        need -= len(got)
    return np.concatenate(out)


_COMP = None


def components():
    global _COMP
    if _COMP is None:
        from hydra import compose, initialize_config_module

        from bio_sbi.pipeline import build_components

        with initialize_config_module("bio_sbi.conf", version_base=None):
            cfg = compose("config", overrides=["+experiment=exact_gw_data_size", "device=cpu"])
        comp = build_components(cfg)
        _COMP = (comp.simulator, comp.generator.observation)
    return _COMP


def theta_from_P(lik, P):
    """Stick coordinates of the outcome matrix P (inverse of the stick map)."""
    th = []
    for r in lik.space.rows:
        if not r.n_sticks:
            continue
        k = lik.sim.law.types.index(r.prefix)
        rem = r.mass
        for j in range(r.n_sticks):
            p = P[k, list(r.free)[j]]
            th.append(p / rem if rem > 0 else 0.5)
            rem -= p
    th = np.clip(np.array(th), 1e-9, 1 - 1e-9)
    assert np.allclose(lik.probabilities(th), P, atol=1e-6), "stick inverse"
    return th


def fit_one(args):
    """MLE of one simulated dataset; returns the 12 probabilities at the MLE."""
    import os

    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    from scipy.optimize import minimize

    from bio_sbi.inference.exact_gw import ExactLikelihood
    from bio_sbi.summaries import Cohort

    p12, design, seed = args
    sim, obs = components()
    rng = np.random.default_rng(seed)
    liks = []
    for fo in ("A", "B"):
        coh = [Cohort(float(h), simulate(p12, fo, h, n, rng)) for (f, h), n in design.items()
               if f == fo and n > 0]  # fmt: skip
        if not coh:
            continue
        lk = ExactLikelihood(sim, obs, coh)
        if fo == "B":
            lk.model = replace(lk.model, founder=np.array([0.0, 1.0]))
        liks.append(lk)
    P0 = np.asarray(p12, float).reshape(2, 6)
    th0 = theta_from_P(liks[0], P0)

    def f(th):
        ll, g = 0.0, np.zeros_like(th)
        for lk in liks:
            a, b = lk.log_likelihood(th)
            ll += a
            g += b
        if not np.isfinite(ll):
            return 1e30, np.zeros_like(th)
        return -ll, -g

    r = minimize(f, th0, jac=True, method="L-BFGS-B", bounds=[(1e-9, 1 - 1e-9)] * len(th0),
                 options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-7})  # fmt: skip
    P = liks[0].probabilities(r.x)
    return P.reshape(-1), bool(r.success), int(r.nit)


def mle_check(n_datasets: int = 30, workers: int = 3) -> pd.DataFrame:
    from optimise import PROTOCOLS as protos  # needs cvxpy (uv run --with cvxpy)

    r1 = [k for k in protos if k.startswith("R1")][0]
    cases = [  # (truth, protocol, N total clones)
        ("T3_Bdiv", r1, 1024),
        ("T3_Bdiv", "A+B 1/2/3/4 (50/50)", 1024),
        ("T3_Bdiv", "current: A 4/6/8/10", 4096),
        ("P2", r1, 1024),
    ]
    rows, draws = [], {}
    for gt, proto, N in cases:
        p12 = design_truth(gt)
        w = protos[proto]
        design = {POINTS[i]: int(round(w[i] * N)) for i in np.flatnonzero(w > 0)}
        Ks, c = load(gt)
        K = design_K(Ks, w)
        lam, V = np.linalg.eigh(K)
        v1 = V[:, 0]
        z = np.load(CACHE / f"{gt}.npz")
        U = z["U"]
        t0 = time.time()
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(fit_one, [(p12, design, 1000 * k + 7) for k in range(n_datasets)]))
        est = np.array([o[0] for o in out])
        dev = est - p12[None, :]
        x = dev @ U  # tangent coordinates
        sd_ba = est[:, I_BTOA].std(ddof=1)
        pred_ba = np.sqrt(c @ np.linalg.solve(N * K, c) / PRIOR_PREC)
        sd_w1 = (x @ v1).std(ddof=1)
        pred_w1 = np.sqrt(1 / (N * PRIOR_PREC * lam[0]))
        on_face = int(np.sum(np.any(est < 1e-6, axis=1) & ~np.any(p12 < 1e-6)))
        rows.append({"truth": SHORTNAME[gt], "design": proto, "N": N, "datasets": n_datasets,
                     "B->A truth": p12[I_BTOA], "B->A mean MLE": est[:, I_BTOA].mean(),
                     "sd B->A MLE": sd_ba, "sd B->A predicted": pred_ba,
                     "ratio sd": sd_ba / pred_ba,
                     "sd w1 MLE": sd_w1, "sd w1 predicted": pred_w1, "ratio sd w1": sd_w1 / pred_w1,
                     "MLEs on a face": on_face,
                     "converged": int(sum(o[1] for o in out)), "seconds": time.time() - t0})  # fmt: skip
        draws[f"{gt} | {proto} | {N}"] = est
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "validation_mle.csv", index=False)
    with open(TAB / "validation_mle.md", "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=".3g"))
    np.savez(OUT / "validation_mle.npz", **{k.replace(" ", "_").replace("|", "").replace("/", "-"): v
                                            for k, v in draws.items()})  # fmt: skip
    return df


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if what in ("posterior", "all"):
        print(posterior_check().groupby("truth")["var ratio post/pred(prior)"].median())
    if what in ("mle", "all"):
        mle_check(nd)
    _ = (BOUNDARY, NAMES)
