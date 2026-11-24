"""Part 3 of the composition-convergence study: does slow mixing make inference easier?

(a) Fisher information (prior-whitened, A-founded clones, XG2 design 4/6/8/10) of the slow
    sets S0/S1/S3, the fast comparator F1, the 9 XG2 truths and the first 60 plausible
    flat-prior draws (``results/exact_gw/optimal_design/info``, same prior stream):
    eigenvalues, 1/lambda of the 4 weakest directions (clones per endpoint at which the data
    match the prior), asymptotic sd of B->A at n = 512, and the per-generation information
    along the weak-4 subspace with its decay rate (vs the Kesten-Stigum rho^-1 and the
    composition-variance rate max(rho^-1, r^2)).
(b) Exact-GW NUTS (+ MLE, ``scripts/exact_gw_mle.py``) at n = 512, one dataset each
    (``OUT/nuts/<set>/n512_s0``), next to the XG2 n = 512 seed-0 runs of the 9 truths.

Writes ``docs/convergence/tables/{info,nuts}.{csv,md}``, ``figures/slow_vs_fast.png``.

  python scripts/convergence/analyse_slow_fast.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    FIG,
    NAMES,
    OUT,
    RESULTS,
    SHORTNAME,
    TAB,
    TRUTHS,
    mean_matrix,
    n_mean,  # fmt: skip
    prior_draws,
    spectrum,
)
from sets import SETS, SLOW  # noqa: E402

OD_INFO = RESULTS / "optimal_design" / "info"
DESIGN = (4, 6, 8, 10)
N_CLONES = 512
PRIOR_PREC = 42.0
I_BTOA = NAMES.index("B.to_A")


def load_info(label: str) -> dict | None:
    """``{gens, K (H, k, k) A-founded whitened, p12, U}``."""
    if label in SETS or label.startswith("mix"):
        f = OUT / "info" / f"{label}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        return {"gens": z["gens"], "K": z["K"], "p12": z["p12"], "U": z["U"], "q": z["q"]}
    z = np.load(OD_INFO / f"{label}.npz")
    return {"gens": z["gens"], "K": z["K"][0], "p12": z["p12"], "U": z["U"], "q": z["q"][0]}


def complete_info(p12: np.ndarray, U: np.ndarray, q: dict) -> np.ndarray:
    """Whitened information if every division outcome of the clone were observed (design
    4/6/8/10, one A-founded detected clone per endpoint): sum_h E[# type-k parents before h |
    detected] x multinomial Fisher of row k (diag(1 / p) on the tangent space), with
    E[. | detected] ~ E[.] / q_h (slight overestimate: extinct clones' events dropped).
    """
    p = np.asarray(p12, float)
    M = mean_matrix(p)
    total = np.zeros(2)
    for h in DESIGN:
        v, cum = np.array([1.0, 0.0]), np.zeros(2)
        for _ in range(h):
            cum += v
            v = v @ M
        total += cum / q[h]
    w = np.concatenate([np.full(6, total[0]), np.full(6, total[1])]) / p
    return U.T @ np.diag(w) @ U / PRIOR_PREC


def info_row(label: str, d: dict, p12_true: np.ndarray) -> tuple[dict, pd.DataFrame]:
    gens = list(map(int, d["gens"]))
    K = d["K"]
    Kd = sum(K[gens.index(h)] for h in DESIGN)  # per clone per endpoint, whitened
    # information retained relative to observing every division (scale-free: removes the
    # 1/p growth of the Fisher information near the simplex boundary)
    Kc = complete_info(d["p12"], d["U"], {h: float(d["q"][gens.index(h)]) for h in DESIGN})
    Lc = np.linalg.cholesky(Kc)
    Li = np.linalg.inv(Lc)
    ret = np.linalg.eigvalsh(Li @ Kd @ Li.T)
    lam, V = np.linalg.eigh(Kd)
    W = V[:, :4]  # weak-4 subspace
    c = d["U"].T @ np.eye(12)[I_BTOA]
    var_asym = float(c @ np.linalg.solve(N_CLONES * PRIOR_PREC * Kd, c))
    # c = A daughters per B cell = B.to_A + 2 B.div_AA + B.div_AB (the dedifferentiation output)
    ec = np.zeros(12)
    ec[[NAMES.index("B.to_A"), NAMES.index("B.div_AA"), NAMES.index("B.div_AB")]] = [1, 2, 1]
    cc = d["U"].T @ ec
    var_c = float(cc @ np.linalg.solve(N_CLONES * PRIOR_PREC * Kd, cc))
    var_prior = float(c @ np.linalg.solve(PRIOR_PREC * (np.eye(len(c)) + N_CLONES * Kd), c))
    # parent-swap subspace (5 dims): an outcome change e (row contrast) taken from A parents
    # and given to B parents in proportion to the stable composition, (-u_B e, u_A e): leaves
    # u^T M (hence rho, u) unchanged, so no growth-rate information leaks into it
    uA = spectrum(mean_matrix(d["p12"]))["u_A"]
    E = np.linalg.qr(np.eye(6) - 1 / 6)[0][:, :5]
    S = np.vstack([-(1 - uA) * E, uA * E])
    Q = np.linalg.qr(d["U"].T @ S)[0]
    per_h = []
    for h in gens:
        Kh = K[gens.index(h)]
        per_h.append({"set": label, "h": h, "weak4_info": float(np.trace(W.T @ Kh @ W)),
                      "swap_info": float(np.trace(Q.T @ Kh @ Q)) / 5,
                      "weak1_info": float(V[:, 0] @ Kh @ V[:, 0]),
                      "btoa_info": float(1 / (c @ np.linalg.pinv(Kh) @ c))
                      if np.linalg.matrix_rank(Kh) == len(c) else np.nan})  # fmt: skip
    ph = pd.DataFrame(per_h)
    # h = 4..10 as report Section 6.3 (beyond ~9 the slow sets' weak-4 trace picks up
    # growth information that grows with h; the parent-swap trace "swap_info" does not decay
    # like the composition variance: swaps also move the offspring variances)
    sel = ph[(ph["h"] >= 4) & (ph["h"] <= 10)]
    decay = float(np.exp(-np.polyfit(sel["h"], np.log(sel["weak4_info"]), 1)[0]))
    M = mean_matrix(p12_true)
    s = spectrum(M)
    late = float(np.trace(W.T @ (K[gens.index(8)] + K[gens.index(10)]) @ W)
                 / np.trace(W.T @ Kd @ W))  # fmt: skip
    row = {"set": label, "rho": s["rho"], "lambda2": s["lambda2"], "r": s["r"],
           "N_mean_01": n_mean(M, 0.01), "coupling": s["coupling"], "a_minus_d": s["a_minus_d"],
           **{f"inv_lam{k + 1}": 1 / lam[k] for k in range(4)},
           "lam_max": lam[-1], "btoa_sd_asym_512": np.sqrt(var_asym),
           "btoa_sd_prior_512": np.sqrt(var_prior),
           "btoa_rel_sd_512": np.sqrt(var_asym) / max(p12_true[I_BTOA], 1e-12),
           "c_sd_asym_512": np.sqrt(var_c), "c_rel_sd_512": np.sqrt(var_c) / s["c"],
           "weak4_decay_per_gen": decay, "rho_KS": s["rho"],
           "comp_var_rate": 1 / max(1 / s["rho"], s["r"] ** 2),
           "weak4_share_h8_h10": late, "retained_min": ret[0], "retained_4th": ret[3],
           "btoa_retained": float(c @ np.linalg.solve(Kc, c)) / float(c @ np.linalg.solve(Kd, c)),
           "btoa_truth": float(p12_true[I_BTOA])}  # fmt: skip
    return row, ph


def info_part() -> tuple[pd.DataFrame, pd.DataFrame]:
    from common import truth_xg2

    rows, perh = [], []
    labels = [(gt, truth_xg2(gt)) for gt in TRUTHS] + list(SETS.items())
    draws, _ = prior_draws(60)
    labels += [(f"prior{i:03d}", draws[i]) for i in range(60)]
    from compute_info import family

    labels += list(family().items())
    for lab, p in labels:
        d = load_info(lab)
        if d is None:
            continue
        r, ph = info_row(lab, d, p)
        r["group"] = ("slow" if lab in SLOW else "fast (F1)" if lab in SETS
                      else "prior" if lab.startswith("prior")
                      else "S1→F1 path" if lab.startswith("mix") else "XG2")  # fmt: skip
        rows.append(r)
        perh.append(ph)
    df = pd.DataFrame(rows)
    ph = pd.concat(perh)
    df.to_csv(TAB / "info.csv", index=False)
    ph.to_csv(TAB / "info_by_generation.csv", index=False)
    show = df[df["group"] != "prior"].copy()
    show["btoa_truth"] = show["btoa_truth"].round(3)
    show["set"] = show["set"].map(lambda x: SHORTNAME.get(x, x))
    cols = ["set", "btoa_truth", "rho", "r", "N_mean_01", "inv_lam1", "inv_lam2", "inv_lam3", "inv_lam4",
            "btoa_sd_asym_512", "btoa_rel_sd_512", "c_sd_asym_512", "c_rel_sd_512",
            "btoa_sd_prior_512", "weak4_decay_per_gen", "rho_KS",
            "comp_var_rate", "weak4_share_h8_h10", "retained_min", "retained_4th",
            "btoa_retained"]  # fmt: skip
    show = show[cols].copy()
    for k in ("retained_min", "retained_4th", "btoa_retained"):
        show[k] = (100 * show[k]).map(lambda v: f"{v:.3g}%")
    md = show.round(3).to_markdown(index=False)
    pr = df[df["group"] == "prior"]
    from scipy.stats import spearmanr

    corr = {y: spearmanr(pr["N_mean_01"], pr[y])[0]
            for y in ("inv_lam1", "btoa_sd_asym_512", "btoa_rel_sd_512", "c_rel_sd_512",
                      "weak4_decay_per_gen", "retained_min",
                      "btoa_retained")}  # fmt: skip
    md += (
        "\n\nFirst 60 plausible prior draws: Spearman(N_mean_01, .) = "
        + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items())
        + "\n"
    )
    (TAB / "info.md").write_text(md + "\n")
    print(md)
    return df, ph


# ---------------------------------------------------------------------- NUTS + MLE
def run_summary(path: Path, label: str, group: str) -> dict | None:
    if not (path / "metrics.json").exists():
        return None
    rec = pd.read_csv(path / "recovery.csv").set_index("parameter")
    draws = pd.read_csv(path / "posterior_samples.csv.gz")[NAMES]
    truth = rec.loc[NAMES, "truth"].to_numpy()
    mean = draws.mean().to_numpy()
    q05, q95 = draws.quantile(0.05).to_numpy(), draws.quantile(0.95).to_numpy()
    q025, q975 = draws.quantile(0.025).to_numpy(), draws.quantile(0.975).to_numpy()
    sd = draws.std().to_numpy()
    met = json.loads((path / "metrics.json").read_text())
    ex = met.get("exact_gw", {})
    stop = ex.get("stop") or {}
    out = {"set": label, "group": group, "btoa_truth": truth[I_BTOA],
           "btoa_mean": mean[I_BTOA], "btoa_sd": sd[I_BTOA],
           "btoa_err": mean[I_BTOA] - truth[I_BTOA],
           "btoa_z": (mean[I_BTOA] - truth[I_BTOA]) / sd[I_BTOA],
           "btoa_q05": q05[I_BTOA], "btoa_q95": q95[I_BTOA],
           "btoa_cov90": bool(q05[I_BTOA] <= truth[I_BTOA] <= q95[I_BTOA]),
           "all12_mae": float(np.mean(np.abs(mean - truth))),
           "all12_cov90": float(np.mean((q05 <= truth) & (truth <= q95))),
           "all12_cov95": float(np.mean((q025 <= truth) & (truth <= q975))),
           "max_rhat": ex.get("max_rhat"), "min_ess": ex.get("min_ess"),
           "divergences": ex.get("divergences"), "stop_reason": stop.get("reason"),
           "draws_per_chain": stop.get("draws_per_chain", ex.get("draws_per_chain")),
           "minutes": met.get("total_seconds", np.nan) / 60}  # fmt: skip
    mf = path / "mle.json"
    if mf.exists():
        m = json.loads(mf.read_text())
        mle = np.array([m["mle"][k] for k in NAMES])
        out["btoa_mle"] = mle[I_BTOA]
        out["btoa_mle_err"] = mle[I_BTOA] - truth[I_BTOA]
        out["all12_mae_mle"] = float(np.mean(np.abs(mle - truth)))
        se = (m.get("se") or {}).get("B.to_A")
        out["btoa_mle_se"] = se if se is not None else np.nan
    return out


def nuts_part() -> pd.DataFrame:
    rows = []
    for seed in (0, 1, 2):
        for lab in SETS:
            r = run_summary(OUT / "nuts" / lab / f"n{N_CLONES}_s{seed}", lab,
                            "slow" if lab in SLOW else "fast (F1)")  # fmt: skip
            if r:
                rows.append(r | {"seed": seed})
        for gt in TRUTHS:
            r = run_summary(RESULTS / "data_size" / gt / f"n{N_CLONES}_s{seed}", SHORTNAME[gt],
                            "XG2")  # fmt: skip
            if r:
                rows.append(r | {"seed": seed})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.to_csv(TAB / "nuts.csv", index=False)
    cols = [c for c in ["set", "seed", "group", "btoa_truth", "btoa_mean", "btoa_sd", "btoa_err",
                        "btoa_z", "btoa_cov90", "btoa_mle", "btoa_mle_err", "btoa_mle_se",
                        "all12_mae", "all12_mae_mle", "all12_cov90", "all12_cov95", "max_rhat",
                        "min_ess", "divergences", "stop_reason", "draws_per_chain", "minutes"]
            if c in df.columns]  # fmt: skip
    md = df[df["group"] != "XG2"][cols].round(3).to_markdown(index=False)
    agg = {
        "btoa_truth": "mean",
        "btoa_sd": "mean",
        "btoa_err": lambda x: np.mean(np.abs(x)),
        "btoa_z": lambda x: np.mean(np.abs(x)),
        "btoa_cov90": "mean",
        "all12_mae": "mean",
        "all12_cov90": "mean",
        "max_rhat": "max",
        "minutes": "mean",
        "seed": "count",
    }
    if "btoa_mle_err" in df.columns:
        agg |= {"btoa_mle_err": lambda x: np.nanmean(np.abs(x)), "all12_mae_mle": "mean"}
    g = df.groupby("set", sort=False).agg(agg).rename(columns={
        "btoa_err": "btoa_abs_err", "btoa_z": "btoa_abs_z", "btoa_mle_err": "btoa_mle_abs_err",
        "seed": "datasets"})  # fmt: skip
    g.to_csv(TAB / "nuts_by_set.csv")
    md += "\n\nMean over datasets (XG2: the n = 512 runs of report Section 7):\n\n"
    md += g.round(3).to_markdown()
    (TAB / "nuts.md").write_text(md + "\n")
    print(md)
    return df


GROUPS = (("prior", "o", "0.6"), ("XG2", "^", "k"), ("slow", "*", "C3"), ("fast (F1)", "s", "C0"))


def _scatter(ax, info: pd.DataFrame, y: str) -> None:
    for g, m, c in GROUPS:
        d = info[info["group"] == g]
        ax.scatter(d["N_mean_01"], d[y], marker=m, color=c, s=50 if g != "prior" else 14,
                   label=g if g != "prior" else "plausible prior draws (first 60)",
                   zorder=3 if g != "prior" else 1)  # fmt: skip
        if g != "prior":
            for _, r in d.iterrows():
                ax.annotate(SHORTNAME.get(r["set"], r["set"]), (r["N_mean_01"], r[y]),
                            xytext=(3, 3), textcoords="offset points", fontsize=7)  # fmt: skip
    path = info[info["group"] == "S1→F1 path"].sort_values("N_mean_01")
    ax.plot(path["N_mean_01"], path[y], "-", color="C2", lw=1.5, label="S1→F1 path (B row mixed)")
    ax.set_yscale("log")
    ax.set_xlabel("N_mean: generation from which pooled A fraction stays within 0.01 of u_A")


def fig(info: pd.DataFrame, ph: pd.DataFrame, nuts: pd.DataFrame) -> None:
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    ax = axs[0, 0]
    col = {"S0": "C3", "S1": "C1", "S3": "C5", "F1": "C0"}
    for lab in TRUTHS:
        d = ph[ph["set"] == lab]
        ax.semilogy(d["h"], d["weak4_info"], "-", color="0.75", lw=0.8)
    for lab in SETS:
        d = ph[ph["set"] == lab]
        if d.empty:
            continue
        r = info.loc[info["set"] == lab].iloc[0]
        ax.semilogy(d["h"], d["weak4_info"], "o-", color=col[lab], ms=3,
                    label=f"{lab}: decay ×{r['weak4_decay_per_gen']:.2f}/gen "
                          f"(ρ={r['rho']:.2f}, 1/max(ρ⁻¹,r²)={r['comp_var_rate']:.2f})")  # fmt: skip
        h0 = 8
        y0 = float(d.loc[d["h"] == h0, "weak4_info"].iloc[0])
        hh = np.arange(3, 13)
        ax.semilogy(hh, y0 * r["rho"] ** -(hh - h0), ":", color=col[lab], lw=0.9)
    ax.plot([], [], "-", color="0.75", label="XG2 truths")
    ax.plot([], [], ":", color="k", label="ρ^-h (Kesten–Stigum)")
    ax.set(xlabel="generation h (one detected A-founded clone)",
           ylabel="information per clone in the weak-4 subspace (whitened)",
           title="(a) weak-direction information per clone vs h (fit h = 4-10)")  # fmt: skip
    ax.legend(fontsize=7)
    ax = axs[0, 1]
    _scatter(ax, info, "inv_lam1")
    ax.set(ylabel="1/λ_min: clones per endpoint to match the prior",
           title="(b) weakest direction, design A-founded 4/6/8/10")  # fmt: skip
    ax.legend(fontsize=7)
    ax = axs[0, 2]
    _scatter(ax, info, "btoa_sd_asym_512")
    ax.set(ylabel="asymptotic sd of B→A, n = 512 per endpoint",
           title="(c) B→A precision (Fisher)")  # fmt: skip
    ax = axs[1, 0]
    _scatter(ax, info, "btoa_retained")
    ax.set(ylabel="B→A information retained vs observing every division",
           title="(d) scale-free: share of complete-data information on B→A")  # fmt: skip
    if not nuts.empty:
        nn = nuts[(nuts["group"] != "XG2") | (nuts["seed"] == 0)].copy()
        nn["set"] = [
            f"{a} s{b}" if g != "XG2" else a
            for a, b, g in zip(nn["set"], nn["seed"], nn["group"], strict=True)
        ]
        nn = nn.reset_index(drop=True)
        colors = [{"slow": "C3", "fast (F1)": "C0"}.get(g, "k") for g in nn["group"]]
        ax = axs[1, 1]
        for i, r in nn.iterrows():
            ax.plot([r["btoa_q05"] - r["btoa_truth"], r["btoa_q95"] - r["btoa_truth"]], [i, i],
                    "-", color=colors[i], lw=2)  # fmt: skip
            ax.plot(r["btoa_err"], i, "o", color=colors[i])
            if np.isfinite(r.get("btoa_mle_err", np.nan)):
                ax.plot(r["btoa_mle_err"], i, "x", color=colors[i], ms=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(nn)))
        ax.set_yticklabels([f"{s} (B→A={t:.2f})" for s, t in
                            zip(nn["set"], nn["btoa_truth"], strict=True)], fontsize=8)  # fmt: skip
        ax.invert_yaxis()
        ax.plot([], [], "o-", color="k", label="posterior mean, 90% interval")
        ax.plot([], [], "x", color="k", label="MLE")
        ax.legend(fontsize=8)
        ax.set(xlabel="B→A estimate − truth (n = 512 per endpoint, dataset 0)",
               title="(e) exact-GW NUTS: slow (red), fast F1 (blue), XG2 (black)")  # fmt: skip
        ax = axs[1, 2]
        x = np.arange(len(nn))
        ax.bar(x - 0.2, nn["all12_mae"], 0.4, color=colors, alpha=0.8, label="posterior mean")
        if "all12_mae_mle" in nn.columns:
            ax.bar(x + 0.2, nn["all12_mae_mle"], 0.4, color=colors, alpha=0.35, hatch="//",
                   label="MLE")  # fmt: skip
        for i, r in nn.iterrows():
            ax.annotate(f"{r['all12_cov90']:.2f}", (i - 0.2, r["all12_mae"]), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7)  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(nn["set"], fontsize=8)
        ax.set(ylabel="mean |estimate − truth| over the 12 probabilities",
               title="(f) all-12 error (numbers: 90% coverage of the 12)")  # fmt: skip
        ax.legend(fontsize=8)
    for a in axs.flat:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "slow_vs_fast.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    info, ph = info_part()
    nuts = nuts_part()
    fig(info, ph, nuts)


if __name__ == "__main__":
    main()
