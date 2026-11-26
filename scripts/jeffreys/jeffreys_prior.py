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


def implied(sim, phys: np.ndarray, model=None) -> dict[str, np.ndarray]:
    """Derived quantities per draw: B division mass, growth rate, mean cells at gen 10."""
    names = list(sim.parameter_names)
    off = np.asarray(sim.law.offspring, float)  # (2, O, 2)
    P = np.stack(
        [phys[:, [names.index(f"{t}.{e}") for e in sim.law.outcomes[t]]] for t in ("A", "B")], 1
    )  # fmt: skip  (outcome order = offspring order)
    M = np.einsum("nko,koj->nkj", P[:, :, : off.shape[1]], off)  # mean offspring matrix
    growth = np.abs(np.linalg.eigvals(M)).max(1)
    m10 = np.array([np.linalg.matrix_power(m, 10)[0].sum() for m in M])
    bdiv = sum(phys[:, names.index(f"B.{e}")] for e in ("div_AA", "div_AB", "div_BB"))
    out = {"B.div": bdiv, "growth": growth, "log10_mean_cells10": np.log10(m10 + 1e-300)}
    if model is not None:  # P(clone detected at generation 10)
        from bio_sbi.inference.gw_exact import detection_and_grad

        z = np.zeros((0, P.shape[2]))
        r = np.zeros(0, np.int64)
        q = [detection_and_grad(model, np.ascontiguousarray(p), z, r, 10)[0] for p in P]
        out["log10_detect10"] = np.log10(np.maximum(q, 1e-300))
    return out


def run_report(burn: float, thin: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.special import expit

    from bio_sbi.inference.smc import ess_bulk, rhat

    prior = jeffreys_prior()
    sim = prior.lik.sim
    z = np.load(OUT / "mcmc.npz")
    U = z["U"]  # (iters + 1, chains, d)
    start = int(burn * len(U))
    kept = U[start::thin]  # (draws, chains, d)
    theta = expit(kept)
    names = list(sim.parameter_names)
    chains = np.transpose(theta, (1, 0, 2))  # (chains, draws, d)
    diag = {"iterations": int(len(U) - 1), "burn": start, "thin": thin,
            "acceptance": float(z["acc"][start:].mean()),
            "max_rhat": float(max(rhat(chains[..., i]) for i in range(theta.shape[2]))),
            "min_ess_bulk": float(min(ess_bulk(chains[..., i]) for i in range(theta.shape[2]))),
            "log_pj_quantiles": np.percentile(z["log_pj"][start:], [1, 50, 99]).tolist()}  # fmt: skip
    flat = theta.reshape(-1, theta.shape[2])
    phys_j = np.array([sim.decode(t) for t in flat])
    rng = np.random.default_rng(1)
    arms = {"jeffreys": phys_j}
    for alpha, key in ((1.0, "flat"), (0.5, "dir_half")):
        a, b = prior.stick_beta(alpha)
        th = np.clip(rng.beta(a, b, size=(len(flat), len(a))), 1e-15, 1 - 1e-15)
        arms[key] = np.array([sim.decode(t) for t in th])
    # flat-B->A: A row flat, B row Dirichlet(to_A 1, others 0.2)
    alpha_b = np.array([0.2 if e != "to_A" else 1.0 for e in
                        ("death", "to_A", "persist", "div_AA", "div_AB", "div_BB")])  # fmt: skip
    pb = rng.dirichlet(alpha_b, len(flat))
    fb = arms["flat"].copy()
    for j, e in enumerate(("death", "to_A", "persist", "div_AA", "div_AB", "div_BB")):
        fb[:, names.index(f"B.{e}")] = pb[:, j]
    arms["flat_btoa"] = fb
    free = list(sim.space.free_names)
    rows = []
    vals = {}
    for arm, ph in arms.items():
        d = {n: ph[:, names.index(n)] for n in free}
        d.update(implied(sim, ph, prior.model))
        vals[arm] = d
        for k, x in d.items():
            q = np.percentile(x, [5, 50, 95])
            rows.append({"arm": arm, "quantity": k, "mean": float(np.mean(x)),
                         "q05": q[0], "median": q[1], "q95": q[2]})  # fmt: skip
    tab = pd.DataFrame(rows)
    ld = vals["jeffreys"]["log10_detect10"]
    diag["jeffreys_frac_detect10_below"] = {
        f"1e{e}": float(np.mean(ld < e)) for e in (-2, -3, -4, -5)
    }
    tab.to_csv(OUT / "prior_marginals.csv", index=False)
    pd.DataFrame(phys_j, columns=names).to_csv(OUT / "jeffreys_prior_draws.csv.gz", index=False)
    (OUT / "mcmc_diagnostics.json").write_text(json.dumps(diag, indent=1))
    print(json.dumps(diag, indent=1))
    print(tab.pivot(index="quantity", columns="arm", values="median").round(3))
    show = ["B.to_A", "B.div", "B.div_BB", "B.death", "A.death", "A.div_AA", "A.to_B",
            "growth", "log10_mean_cells10", "log10_detect10"]  # fmt: skip
    colors = {"flat": "#4C72B0", "flat_btoa": "#DD8452", "dir_half": "#55A868",
              "jeffreys": "#C44E52"}  # fmt: skip
    label = {"flat": "flat Dirichlet(1)", "flat_btoa": "flat-B→A", "dir_half": "Dirichlet(1/2)",
             "jeffreys": "exact Jeffreys"}  # fmt: skip
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
    for ax, k in zip(axes.flat, show, strict=True):
        lo = min(np.percentile(vals[a][k], 0.5) for a in vals)
        hi = max(np.percentile(vals[a][k], 99.5) for a in vals)
        bins = np.linspace(lo, hi, 41)
        for arm in ("flat", "flat_btoa", "dir_half", "jeffreys"):
            ax.hist(vals[arm][k], bins=bins, density=True, histtype="step", lw=1.6,
                    color=colors[arm], label=label[arm])  # fmt: skip
        ax.set_title(k)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Prior marginals (exact Jeffreys: DE-MCMC draws)")
    fig.tight_layout()
    fig.savefig(OUT / "prior_marginals.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["is", "profiles", "summary", "report"])
    ap.add_argument("--burn", type=float, default=0.33)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()
    if args.cmd == "is":
        run_is(args.n, args.alpha, args.seed, args.chunk)
    elif args.cmd == "profiles":
        run_profiles()
    elif args.cmd == "report":
        run_report(args.burn, args.thin)
    else:
        run_summary(f"is_dir{args.alpha:g}_s{args.seed}", args.seed)


if __name__ == "__main__":
    main()
