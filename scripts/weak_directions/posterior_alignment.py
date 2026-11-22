"""Check the Fisher weak directions against the XG2 NUTS posteriors (flat prior).

For every XG2 run at n = 256, 512, 1024 (and 2048 / 4096 for T1-T4), seeds 0-2: the
posterior covariance of the draws in the prior-whitened tangent coordinates (x 42) is
compared with the Gaussian prediction (I + n K)^-1 from the exact Fisher information
(K = I / 42, design 4/6/8/10) at the truth and at the run's posterior mean:

* principal directions: |cos| between the posterior's top PCs and the Fisher weakest
  eigenvectors, mean cos^2 of the principal angles between the 2- and 4-dim subspaces;
* variances along the Fisher eigenvectors vs 1 / (1 + n lam);
* shape along the weakest direction (skewness, excess kurtosis, bimodality coefficient)
  and the posterior mass with B dividing (sum of B divisions > 0.15) for boundary truths.

Writes ``tables/posterior_alignment.{csv,md}`` and ``figures/posterior_alignment.png``.

  uv run python scripts/weak_directions/posterior_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from analyse_fisher import TRUTH_SHORT, _style  # noqa: E402
from common import (  # noqa: E402
    FIG, HORIZONS, NAMES, OUT, PRIOR_PREC, TAB, TRUTHS, XG2, eig_tangent, information_12,
    label, row_contrasts, truth,
)  # fmt: skip
from scipy import stats  # noqa: E402

NS = (256, 512, 1024, 2048, 4096)
SEEDS = (0, 1, 2)
NW = 4


def draws(gt: str, n: int, s: int) -> np.ndarray | None:
    f = XG2 / gt / f"n{n}_s{s}" / "posterior_samples.csv.gz"
    if not f.exists():
        return None
    return pd.read_csv(f)[NAMES].to_numpy()


def info_at(p12: np.ndarray, key: str) -> np.ndarray:
    """Base-design information (one clone per endpoint) at p12, cached."""
    f = OUT / "info_postmean.npz"
    cache = dict(np.load(f)) if f.exists() else {}
    if key in cache and np.allclose(cache[key + "_p"], p12):
        return cache[key]
    info = sum(information_12(p12, g)[0] for g in HORIZONS)
    cache[key], cache[key + "_p"] = info, p12
    np.savez_compressed(f, **cache)
    return info


def bimodality(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient (> 5/9 ~ 0.555 suggests bi-/multimodality)."""
    g, k = stats.skew(x), stats.kurtosis(x)
    m = len(x)
    return float((g**2 + 1) / (k + 3 * (m - 1) ** 2 / ((m - 2) * (m - 3))))


def compare(Sc: np.ndarray, Kc: np.ndarray, n: int) -> dict:
    """Posterior (whitened, tangent coords) vs Gaussian prediction from K."""
    lam, c = np.linalg.eigh(Kc)
    pv, pc = np.linalg.eigh(Sc)
    pv, pc = pv[::-1], pc[:, ::-1]  # largest posterior variance first
    out = {}
    for k in range(NW):
        out[f"cos_pc{k + 1}_w{k + 1}"] = float(abs(pc[:, k] @ c[:, k]))
        out[f"best_cos_w{k + 1}"] = float(np.max(np.abs(pc[:, :NW].T @ c[:, k])))
        out[f"var_ratio_w{k + 1}"] = float(c[:, k] @ Sc @ c[:, k] * (1 + n * lam[k]))
    for m in (1, 2, 4):
        s = np.linalg.svd(pc[:, :m].T @ c[:, :m], compute_uv=False)
        out[f"overlap{m}"] = float(np.mean(s**2))
    out["post_var_top4"] = pv[:NW].tolist()
    out["pred_var_top4"] = (1 / (1 + n * lam[:NW])).tolist()
    return out


def main() -> None:
    U = row_contrasts()
    z = np.load(OUT / "fisher_eigen.npz")
    rows, examples = [], {}
    for gt in TRUTHS:
        K = z[f"{gt}_K"]
        Kc = U.T @ K @ U
        V = z[f"{gt}_V"]
        for n in NS:
            for s in SEEDS:
                X = draws(gt, n, s)
                if X is None:
                    continue
                Sc = U.T @ np.cov(X.T) @ U * PRIOR_PREC
                r = {"truth": gt, "n": n, "seed": s, "draws": len(X)}
                r.update({f"truth_{k}": v for k, v in compare(Sc, Kc, n).items()})
                pm = X.mean(0)
                Kp = info_at(pm, f"{gt}_n{n}_s{s}") / PRIOR_PREC
                r.update({f"pmean_{k}": v for k, v in compare(Sc, U.T @ Kp @ U, n).items()})
                lp, Vp = eig_tangent(Kp)
                r["pmean_lam1"], r["pmean_w1_label"] = lp[0], label(Vp[:, 0], 5)
                r["pmean_cos_w1_truth_w1"] = float(abs(Vp[:, 0] @ V[:, 0]))
                proj = (X - X.mean(0)) @ V[:, 0]
                r["w1_skew"] = float(stats.skew(proj))
                r["w1_exkurt"] = float(stats.kurtosis(proj))
                r["w1_bimodality"] = bimodality(proj)
                bdiv = X[:, 9:12].sum(1)
                r["mass_B_divides_gt_0.15"] = float(np.mean(bdiv > 0.15))
                ib = NAMES.index("B.to_A")
                lam = z[f"{gt}_lam"]
                r["sd_BtoA_post"] = float(X[:, ib].std())
                r["sd_BtoA_gauss_truth"] = float(np.sqrt(
                    np.sum(V[ib] ** 2 / (1 + n * lam)) / PRIOR_PREC))  # fmt: skip
                rows.append(r)
                if s == 0 and n == 1024:
                    examples[gt] = (X, V, lam)
                print(gt, n, s, f"overlap4 {r['truth_overlap4']:.2f} "
                      f"vr1 {r['truth_var_ratio_w1']:.2f}", flush=True)  # fmt: skip
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in df.columns if c.endswith("top4")]).to_csv(
        TAB / "posterior_alignment.csv", index=False)
    summ = (df.groupby(["truth", "n"], sort=False)
            [["truth_overlap1", "truth_overlap2", "truth_overlap4", "pmean_overlap4",
              "truth_cos_pc1_w1", "truth_best_cos_w1", "truth_var_ratio_w1",
              "truth_var_ratio_w2", "truth_var_ratio_w3", "truth_var_ratio_w4",
              "pmean_var_ratio_w1", "w1_bimodality", "mass_B_divides_gt_0.15",
              "sd_BtoA_post", "sd_BtoA_gauss_truth"]].mean().reset_index())  # fmt: skip
    summ.to_csv(TAB / "posterior_alignment_summary.csv", index=False)
    (TAB / "posterior_alignment_summary.md").write_text(
        summ.to_markdown(index=False, floatfmt=".2f") + "\n")
    plot(df, examples)


def plot(df: pd.DataFrame, examples: dict) -> None:
    import matplotlib.pyplot as plt

    _style()
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
            "#e34948", "#52514e"]  # fmt: skip
    fig = plt.figure(figsize=(7.4, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    # overlap of the weak-4 subspaces (truth Fisher) vs n
    ax = fig.add_subplot(gs[0, 0])
    for i, gt in enumerate(TRUTHS):
        s = df[df.truth == gt].groupby("n").truth_overlap4.agg(["mean", "min", "max"])
        ax.plot(s.index, s["mean"], "-o", color=cols[i], ms=3.5, lw=1.2, label=TRUTH_SHORT[gt])
    ax.axhline(0.4, color="#c3c2b7", ls="--", lw=0.8)
    ax.text(270, 0.42, "random", fontsize=6.5, color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("n (clones per endpoint)")
    ax.set_title("posterior top-4 PCs vs Fisher\nweakest 4 (mean cos²)", fontsize=8)
    # first PC vs weakest direction
    ax = fig.add_subplot(gs[0, 1])
    for i, gt in enumerate(TRUTHS):
        s = df[df.truth == gt].groupby("n").truth_best_cos_w1.mean()
        ax.plot(s.index, s.values, "-o", color=cols[i], ms=3.5, lw=1.2)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("n (clones per endpoint)")
    ax.set_title("weakest Fisher direction: best |cos|\nwith a posterior top-4 PC", fontsize=8)
    # variance ratio along weakest direction
    ax = fig.add_subplot(gs[0, 2])
    for i, gt in enumerate(TRUTHS):
        for k, mk in ((1, "o"), (2, "s")):
            s = df[df.truth == gt].groupby("n")[f"truth_var_ratio_w{k}"].mean()
            ax.plot(s.index * (1.06 if k == 2 else 1), s.values, mk, color=cols[i], ms=3.5,
                    mfc=cols[i] if k == 1 else "none")  # fmt: skip
    ax.axhline(1, color="#c3c2b7", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n (clones per endpoint)")
    ax.set_title("posterior variance / Gaussian\nprediction along w1 (●), w2 (○)", fontsize=8)
    h, lab = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=9, fontsize=7, bbox_to_anchor=(0.5, 0.515))
    # examples: draws projected on w1, w2 at n = 1024 (seed 0) with the Gaussian 2-sd ellipse
    for j, gt in enumerate(("T1_ref", "T3_Bdiv", "P3")):
        ax = fig.add_subplot(gs[1, j])
        X, V, lam = examples[gt]
        P = (X - X.mean(0)) @ V[:, :2]
        bd = X[:, 9:12].sum(1)
        sc = ax.scatter(P[:, 0], P[:, 1], s=3, c=bd, cmap="Blues", vmin=0, vmax=0.7,
                        alpha=0.7, lw=0)  # fmt: skip
        tp = (truth(gt) - X.mean(0)) @ V[:, :2]
        ax.plot(tp[0], tp[1], "*", ms=10, color="#eda100", mec="#0b0b0b", mew=0.6, zorder=5)
        t = np.linspace(0, 2 * np.pi, 200)
        sd = 2 / np.sqrt(PRIOR_PREC * (1 + 1024 * lam[:2]))
        ax.plot(sd[0] * np.cos(t), sd[1] * np.sin(t), color="#e34948", lw=1.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("along w1 (probability units)")
        if j == 0:
            ax.set_ylabel("along w2")
        ax.set_title(f"{TRUTH_SHORT[gt]}, n = 1024, dataset 0", fontsize=8)
    cax = fig.add_axes((0.925, 0.1, 0.012, 0.33))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("B division mass", fontsize=7, labelpad=1)
    fig.text(0.01, 0.005, "Red ellipse: 2-sd Gaussian prediction (flat prior + n x Fisher at the "
             "truth), centred at the posterior mean. Points: NUTS draws; star: truth.", fontsize=6.5, color="#52514e")  # fmt: skip
    fig.subplots_adjust(left=0.08, right=0.91, top=0.93, bottom=0.08, hspace=0.62, wspace=0.42)
    fig.savefig(FIG / "posterior_alignment.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
