"""Approximation checks on a subset of runs: evidence of M_full and M0 by every estimator.

Usage: uv run python scripts/spike_slab/validate.py [--workers 4]
Needs evidence_runs.py output for the runs in VALIDATION (and nuts_m0.py output where
available). Adds the Laplace approximations in physical coordinates at each model's MLE
(plain, and truncated to the simplex with the gradient term) and writes
docs/spike_slab/tables/validation_evidence.{csv,md} and validation_m0_posterior.{csv,md}.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from core import DOCS, FREE12, MODELS, OUT, Face, laplace_physical, list_runs, load, run_dir

VALIDATION = [("T1_ref", 128, 0), ("T1_ref", 1024, 0), ("T1_ref", 4096, 0), ("T2_toA05", 128, 1),
              ("T2_toA05", 512, 0), ("T4_slowA", 256, 0), ("T3_Bdiv", 512, 0), ("P1", 128, 0),
              ("P3", 512, 0), ("T5_C7", 512, 0)]  # fmt: skip
NUTS_RUNS = [("T1_ref", 128, 0), ("T1_ref", 1024, 0), ("T2_toA05", 512, 0), ("T3_Bdiv", 512, 0),
             ("P3", 512, 0)]  # fmt: skip


def one(key):
    gt, n, seed = key
    run = next(r[3] for r in list_runs() if r[:3] == key)
    lik, names, truth, post, mle = load(Path(run))
    rows = []
    for model in ("full", "M0"):
        z = np.load(run_dir(*key) / f"{model}.npz")
        face = Face(lik, names, MODELS[model], post[0], ref_phys=post.mean(0))
        lp = laplace_physical(face, np.asarray(z["mle_phys"], dtype=float))
        row = {"gt": gt, "n": n, "seed": seed, "model": model,
               "laplace_u": float(z["logZ_laplace"]),
               "laplace_phys": lp["logZ_laplace_phys"], "laplace_trunc": lp["logZ_laplace_trunc"],
               "laplace_trunc0": lp["logZ_laplace_trunc0"], "trunc0_prob": lp["trunc0_prob"],
               "trunc_prob": lp["trunc_prob"], "grad_term": lp["grad_term"],
               "at_zero": " ".join(a.replace("B.", "").replace("A.", "A:") for a in lp["at_zero"]),
               "is_k4": float(z["logZ_is_k4"]), "pareto_k_k4": float(z["pareto_k_k4"]),
               "smc_internal": float(z["logZ_smc"]) if "logZ_smc" in z.files else np.nan,
               "bridge_k1": float(z["logZ_bridge_k1"]), "bridge_k4": float(z["logZ_bridge_k4"]),
               "bridge": float(z["logZ_main"])}  # fmt: skip
        f = OUT / "nuts" / f"{gt}_n{n}_s{seed}_{model}.npz"
        if model == "M0" and f.exists():
            zn = np.load(f)
            row["bridge_nuts"] = float(np.mean([zn[f"logZ_bridge_k{k}"] for k in (1, 4, 8)]))
        rows.append(row)
    return rows


def m0_posterior_check() -> pd.DataFrame:
    rows = []
    for key in NUTS_RUNS:
        gt, n, seed = key
        f = OUT / "nuts" / f"{gt}_n{n}_s{seed}_M0.npz"
        g = run_dir(*key) / "M0.npz"
        if not (f.exists() and g.exists()):
            continue
        zn, zs = np.load(f), np.load(g)
        names = json.loads((run_dir(*key) / "meta.json").read_text())["names"]
        idx = [names.index(p) for p in FREE12 if not p.startswith("B.div")]
        A = zn["phys"].reshape(-1, zn["phys"].shape[-1])[:, idx].astype(float)
        B = np.asarray(zs["post_phys"], dtype=float)[:, idx]
        for j, i in enumerate(idx):
            rows.append({"gt": gt, "n": n, "seed": seed, "parameter": names[i],
                         "nuts_mean": A[:, j].mean(), "smc_mean": B[:, j].mean(),
                         "nuts_sd": A[:, j].std(), "smc_sd": B[:, j].std(),
                         "nuts_q05": np.quantile(A[:, j], 0.05), "smc_q05": np.quantile(B[:, j], 0.05),
                         "nuts_q95": np.quantile(A[:, j], 0.95), "smc_q95": np.quantile(B[:, j], 0.95),
                         "nuts_rhat": float(zn["rhat"]), "nuts_ess_min": float(zn["ess_min"])})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    keys = [k for k in VALIDATION if (run_dir(*k) / "M0.npz").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for rs in pool.map(one, keys) for r in rs]
    df = pd.DataFrame(rows)
    tab = DOCS / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    df.to_csv(tab / "validation_evidence.csv", index=False)
    # differences to the bridge estimate (the reference)
    show = df.copy()
    for c in ("laplace_u", "laplace_phys", "laplace_trunc0", "laplace_trunc", "is_k4", "smc_internal", "bridge_k1",
              "bridge_k4", "bridge_nuts"):
        if c in show:
            show[c] = show[c] - show["bridge"]
    show["bridge"] = show["bridge"].round(2)
    (tab / "validation_evidence.md").write_text(show.round(2).to_markdown(index=False))
    m0 = m0_posterior_check()
    if len(m0):
        m0.to_csv(tab / "validation_m0_posterior.csv", index=False)
        (tab / "validation_m0_posterior.md").write_text(m0.round(3).to_markdown(index=False))
    print(show.round(2).to_string())
    print(m0.round(3).to_string())


if __name__ == "__main__":
    main()
