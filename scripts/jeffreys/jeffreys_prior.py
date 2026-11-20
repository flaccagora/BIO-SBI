"""What the exact Jeffreys prior of the XG2 model implies, and whether it is proper.

Subcommands (outputs in results/exact_gw/data_size_jeffreys/prior/):

* ``is [--n N] [--alpha A] [--seed S]``: importance sampling of the Jeffreys prior from a
  symmetric Dirichlet(A) proposal on both rows (theta = stick coordinates). Saves draws,
  ``log p_J`` (unnormalised, theta coordinates), ``log q`` and diagnostics in chunks
  (restart-safe). The Pareto k of the weights p_J / q and the running estimate of the
  normaliser are the numerical propriety check (relative to the proposal).
* ``profiles``: ``log p_J`` in physical coordinates (free probabilities of each row)
  along rays to every face (one outcome -> 0) and every vertex (one outcome -> 1) of each
  row from interior base points; the local exponent beta of p_J ~ eps^beta decides
  integrability (face: beta > -1; vertex of a 5-simplex: beta + 4 > -1).
* ``summary``: PSIS weights, normaliser, Pareto k, resampled Jeffreys draws
  (``jeffreys_draws.csv.gz``), marginal summaries vs Dirichlet(1) / Dirichlet(1/2).

  uv run python scripts/jeffreys/jeffreys_prior.py is --n 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JEFF, evaluate, jeffreys_prior  # noqa: E402

OUT = JEFF / "prior"


def run_is(n: int, alpha: float, seed: int, chunk: int) -> None:
    prior = jeffreys_prior()
    a, b = prior.stick_beta(alpha)
    rng = np.random.default_rng(seed)
    theta = rng.beta(a, b, size=(n, len(a)))
    theta = np.clip(theta, 1e-12, 1 - 1e-12)
    tag = f"is_dir{alpha:g}_s{seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(0, n, chunk):
        f = OUT / f"{tag}_{i // chunk:03d}.npz"
        if f.exists():
            continue
        th = theta[i : i + chunk]
        res = evaluate(prior, th, log=None)
        log_q = np.array([prior.log_dirichlet(t, alpha) for t in th])
        np.savez_compressed(f, theta=th, log_q=log_q, **res)
        print(f"{tag} chunk {i // chunk}: {i + len(th)}/{n}", flush=True)


def load_is(tag: str) -> dict[str, np.ndarray]:
    files = sorted(OUT.glob(f"{tag}_[0-9][0-9][0-9].npz"))
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def physical_logdet_jacobian(prior, theta: np.ndarray) -> float:
    """``log |det dP_free / dtheta|`` summed over rows (theta -> free probabilities)."""
    P, dP, rows = prior.lik.jacobian(theta)
    out = 0.0
    for k in (0, 1):
        jac = dP[rows == k][:, : int(prior.model.n_out[k]) - 1]
        out += np.log(abs(np.linalg.det(jac)))
    return out


def run_profiles() -> None:
    prior = jeffreys_prior()
    sim = prior.lik.sim
    names = list(sim.parameter_names)
    bases = {
        "T3_Bdiv": {"A": {"death": 0.03, "persist": 0.03, "to_B": 0.05, "div_AA": 0.55,
                          "div_AB": 0.05, "div_BB": 0.29},
                    "B": {"death": 0.10, "persist": 0.08, "to_A": 0.60, "div_AA": 0.02,
                          "div_AB": 0.05, "div_BB": 0.15}},
        "uniform": {"A": dict.fromkeys(("death", "persist", "to_B", "div_AA", "div_AB",
                                         "div_BB"), 1 / 6),
                    "B": dict.fromkeys(("death", "persist", "to_A", "div_AA", "div_AB",
                                         "div_BB"), 1 / 6)},
    }  # fmt: skip
    eps = 10.0 ** -np.arange(1, 9)
    rows = []
    for bname, base in bases.items():
        phys0 = sim.physical_from_mapping(base)
        for row in ("A", "B"):
            idx = [i for i, nm in enumerate(names) if nm.startswith(row + ".")]
            for o in idx:
                for kind in ("face", "vertex"):
                    thetas = []
                    for e in eps:
                        ph = phys0.copy()
                        p = ph[idx].copy()
                        if kind == "face":  # outcome o -> eps, others rescaled
                            rest = [i for i in range(len(idx)) if idx[i] != o]
                            p[rest] = p[rest] / p[rest].sum() * (1 - e)
                            p[idx.index(o)] = e
                        else:  # outcome o -> 1 - eps, the others share eps
                            rest = [i for i in range(len(idx)) if idx[i] != o]
                            p[rest] = p[rest] / p[rest].sum() * e
                            p[idx.index(o)] = 1 - e
                        ph[idx] = p
                        thetas.append(np.clip(sim.encode(ph), 1e-15, 1 - 1e-15))
                    thetas = np.asarray(thetas)
                    res = evaluate(prior, thetas, log=None)
                    lj = np.array([physical_logdet_jacobian(prior, t) for t in thetas])
                    lp = res["log_pj"] - lj  # density in the free physical probabilities
                    slope = np.diff(lp) / np.diff(np.log(eps))
                    rows.append({
                        "base": bname, "row": row, "outcome": names[o], "kind": kind,
                        "eps": eps.tolist(), "log_pj_phys": lp.tolist(),
                        "slopes": slope.tolist(), "beta_small_eps": float(slope[-3:].mean()),
                        "min_eig": res["min_eig"].tolist(), "omitted": res["omitted"].tolist(),
                    })  # fmt: skip
                    print(bname, names[o], kind, np.round(slope, 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "boundary_profiles.json").write_text(json.dumps(rows, indent=1))


def run_summary(tag: str, seed: int) -> None:
    import pandas as pd

    from bio_sbi.inference.smc import psis

    prior = jeffreys_prior()
    sim = prior.lik.sim
    d = load_is(tag)
    lw = d["log_pj"] - d["log_q"]
    ok = np.isfinite(lw)
    lw_s, k = psis(np.where(ok, lw, -np.inf))
    w = np.exp(lw_s - lw_s[ok].max())
    w[~ok] = 0
    w /= w.sum()
    m = lw[ok].max()
    log_z = m + np.log(np.mean(np.exp(lw[ok] - m)))
    # running normaliser (raw weights) at 1/8 .. 8/8 of the draws
    n = len(lw)
    run_z = []
    for f in np.arange(1, 9) / 8:
        sl = lw[: int(n * f)]
        sl = sl[np.isfinite(sl)]
        run_z.append(float(sl.max() + np.log(np.mean(np.exp(sl - sl.max())))))
    rng = np.random.default_rng(seed)
    pick = rng.choice(n, size=n, replace=True, p=w)
    phys = np.array([sim.decode(t) for t in d["theta"][pick]])
    names = list(sim.parameter_names)
    df = pd.DataFrame(phys, columns=names)
    df.to_csv(OUT / "jeffreys_draws.csv.gz", index=False)
    phys_q = np.array([sim.decode(t) for t in d["theta"]])
    pd.DataFrame(phys_q, columns=names).assign(log_w=lw, w_psis=w).to_csv(
        OUT / f"{tag}_draws.csv.gz", index=False
    )
    info = {
        "tag": tag, "n": int(n), "finite": int(ok.sum()), "pareto_k": float(k),
        "ess": float(1 / np.sum(w**2)), "log_normaliser": float(log_z),
        "running_log_normaliser": run_z, "min_eig_min": float(d["min_eig"].min()),
        "omitted_max": float(d["omitted"].max()), "tail_max": float(np.nanmax(d["tail"])),
        "grid10_max": int(d["grid10"].max()),
        "unique_resampled": int(len(np.unique(pick))),
    }  # fmt: skip
    (OUT / f"{tag}_summary.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["is", "profiles", "summary"])
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()
    if args.cmd == "is":
        run_is(args.n, args.alpha, args.seed, args.chunk)
    elif args.cmd == "profiles":
        run_profiles()
    else:
        run_summary(f"is_dir{args.alpha:g}_s{args.seed}", args.seed)


if __name__ == "__main__":
    main()
