"""Analyse the profile files of profile_runs.py (no likelihood evaluations).

Per run: profile drop across B.to_A in [0.2, 0.9], 95% profile interval (1.92 nats),
marginal posterior density of B.to_A (reflected KDE of the flat-prior draws), the
"volume" term log V(c) = log p(c | y) - [profile(c) - max profile], the mean of B.to_A
under profile-only weights exp(profile) (what the posterior mean would be with no volume
effect) and under exp(profile) * (1 - c)^4 (profile times the flat Dirichlet's marginal
prior Beta(1, 5)), the draws' log-likelihood deficit vs Gamma(d/2), B division mass and
active bounds along the profile, and the likelihood-ratio test of "B never divides".

  uv run python scripts/xg2_mle_vs_posterior/analyse_profiles.py
Writes docs/xg2_mle_vs_posterior/profiles.csv, profile_curves.csv and figures/.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import OUT  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402
from scipy.stats import chi2, gamma, gaussian_kde  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
D = 10  # free parameters
FINE = np.linspace(0.0, 1.0, 1001)


def kde_logpdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    bw = 0.5 * len(x) ** -0.2 / np.sqrt(3)  # Scott-like on the 3x reflected sample
    kde = gaussian_kde(np.r_[x, -x, 2 - x], bw_method=bw)
    return np.log(3 * np.maximum(kde(grid), 1e-300))


def interval(grid: np.ndarray, prof: np.ndarray, drop: float = 1.92) -> tuple[float, float]:
    f = PchipInterpolator(grid, prof)(FINE[(FINE >= grid[0]) & (FINE <= grid[-1])])
    g = FINE[(FINE >= grid[0]) & (FINE <= grid[-1])]
    ok = g[f >= f.max() - drop]
    return float(ok.min()), float(ok.max())


def weighted_mean(grid: np.ndarray, logw: np.ndarray) -> float:
    g = FINE[(FINE >= grid[0]) & (FINE <= grid[-1])]
    lw = PchipInterpolator(grid, logw)(g)
    w = np.exp(lw - lw.max())
    return float((g * w).sum() / w.sum())


def main() -> None:
    files = sorted((OUT / "profiles").glob("*.npz"))
    rows, curves = [], []
    fig_dir = OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        z = np.load(f)
        names = list(z["names"])
        ia = names.index("B.to_A")
        bidx = [names.index(k) for k in ("B.div_AA", "B.div_AB", "B.div_BB")]
        free = [i for i, n in enumerate(names) if n.startswith(("A.", "B."))]
        grid, prof = z["grid"], z["prof_ll"]
        ll_max = max(float(z["ll_mle"]), prof.max())
        dprof = prof - ll_max
        post, truth, dll = z["post"], z["truth"], z["draws_ll"]
        x = post[:, ia]
        logp = kde_logpdf(x, grid)
        logv = logp - dprof
        logv -= logv[np.argmin(np.abs(grid - grid[np.argmax(prof)]))]
        # KDE log density is meaningless where the draws have (almost) no mass
        logv = np.where(logp > logp.max() - 6, logv, np.nan)
        div_prof = z["prof_phys"][:, bidx].sum(1)
        active = (z["prof_phys"][:, free] < 1e-4).sum(1)
        deficit = ll_max - dll
        lo, hi = interval(grid, prof)
        in_range = (grid >= 0.2) & (grid <= 0.9)
        lr = 2 * (ll_max - float(z["nodiv_ll"]))
        rows.append({
            "run": f.stem, "truth": truth[ia], "mle": z["mle_phys"][ia],
            "profile_argmax": grid[np.argmax(prof)],
            "global_mle": grid[np.argmax(prof)] if prof.max() > float(z["ll_mle"]) + 0.01
            else z["mle_phys"][ia],
            "post_mean": x.mean(),
            "post_median": np.median(x),
            "mean_profile_only": weighted_mean(grid, dprof),
            "mean_profile_x_beta15": weighted_mean(grid, dprof + 4 * np.log1p(-grid)),
            "profile_drop_0.2_0.9": float(dprof[in_range].max() - dprof[in_range].min()),
            "profile_at_truth": float(PchipInterpolator(grid, dprof)(truth[ia])),
            "pi95_lo": lo, "pi95_hi": hi,
            "post_q025": np.quantile(x, 0.025), "post_q975": np.quantile(x, 0.975),
            "logV_range": float(np.nanmax(logv) - np.nanmin(logv)),
            "logV_low_minus_high": float(np.interp(0.2, grid, logv) - np.interp(0.8, grid, logv)),
            "ll_mle": ll_max, "ll_mle_stored": float(z["ll_mle_stored"]),
            "max_draw_ll_gap": float(deficit.min()), "mean_deficit": float(deficit.mean()),
            "gamma5_mean": D / 2, "frac_deficit_gt_q95_gamma5": float(
                (deficit > gamma.ppf(0.95, D / 2)).mean()),
            "nodiv_ll_gap": ll_max - float(z["nodiv_ll"]), "LR_nodiv": lr,
            "p_chibar_3": float(sum(w * chi2.sf(lr, k) for k, w in
                                    ((1, 3 / 8), (2, 3 / 8), (3, 1 / 8))) if lr > 0 else 1.0),
            "nodiv_B.to_A": float(z["nodiv_phys"][ia]),
            "div_mass_at_profile_0.2": float(np.interp(0.2, grid, div_prof)),
            "div_mass_at_profile_0.8": float(np.interp(0.8, grid, div_prof)),
            "evals": int(z["evals"]), "seconds": float(z["seconds"]),
        })  # fmt: skip
        for j, c in enumerate(grid):
            curves.append({"run": f.stem, "c": c, "dprofile": dprof[j], "log_post_marg": logp[j],
                           "logV": logv[j], "div_mass": div_prof[j], "active_bounds": active[j],
                           **{f"prof_{names[i]}": z["prof_phys"][j, i] for i in free}})  # fmt: skip

        # figure: profile vs marginal posterior, and what moves along the ridge
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        top = logp.max()
        ax[0].plot(grid, dprof, "-o", ms=3, color=C[0], label="profile log lik (max = 0)")
        ax[0].plot(grid, logp - top, "-", color=C[1], label="log marginal posterior (flat prior)")
        ax[0].plot(grid, 4 * np.log1p(-grid) - 4 * np.log1p(-grid[np.argmax(prof)]), ":",
                   color="0.4", label="4 log(1-c): Beta(1,5) prior, shifted")
        ax[0].axhline(-1.92, color="0.6", lw=0.8)
        for v, ls, lab in ((truth[ia], "-", "truth"), (z["mle_phys"][ia], "--", "MLE"),
                           (x.mean(), "-.", "post. mean")):
            ax[0].axvline(v, color="k", ls=ls, lw=1, label=lab)
        ax[0].set_ylim(max(-15, dprof.min() - 1), 1)
        ax[0].set_xlabel("B.to_A = c")
        ax[0].set_ylabel("nats")
        ax[0].set_title(f.stem)
        ax[0].legend(fontsize=7, loc="lower center")
        ax[1].plot(grid, logv, "-o", ms=3, color=C[2])
        ax[1].set_xlabel("B.to_A = c")
        ax[1].set_ylabel("log V(c) = log marg - profile (0 at peak)")
        ax[1].plot(grid, 4 * np.log1p(-grid) - 4 * np.log1p(-grid[np.argmax(prof)]), ":",
                   color="0.4", label="4 log(1-c) (no information on nuisance)")
        ax[1].set_ylim(np.nanmin(logv) - 1, np.nanmax(logv) + 1)
        ax[1].legend(fontsize=7)
        ax[1].set_title("volume term (where the draws have mass)")
        ax[2].plot(grid, div_prof, "-o", ms=3, color=C[3], label="B division mass")
        for k, nm in enumerate(("B.death", "B.persist")):
            ax[2].plot(grid, z["prof_phys"][:, names.index(nm)], "-", color=C[k], label=nm)
        ax[2].set_xlabel("B.to_A = c")
        ax[2].set_ylabel("probability at profile optimum")
        ax[2].set_title("along the profile")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"profile_{f.stem}.png", dpi=110)
        plt.close(fig)

        # deficit distribution vs Gamma(d/2)
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        ax.hist(deficit, bins=40, density=True, color=C[0], alpha=0.7, label="draws")
        t = np.linspace(0, max(deficit.max(), 15), 300)
        ax.plot(t, gamma.pdf(t, D / 2), color=C[1], label="Gamma(5,1)")
        ax.set_xlabel("max log lik - log lik(draw)")
        ax.set_title(f.stem, fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"deficit_{f.stem}.png", dpi=100)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "profiles.csv", index=False)

    # one dot plot: where each summary of B.to_A lands, per profiled run
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))
    marks = [("truth", "truth", "k", "|"), ("global_mle", "MLE (global)", C[1], "D"),
             ("mean_profile_only", "mean under exp(profile)  [no volume]", C[2], "s"),
             ("post_mean", "posterior mean (flat prior)", C[0], "o"),
             ("mean_profile_x_beta15", "mean under exp(profile) x Beta(1,5)", C[4], "v")]
    for j, r in df.reset_index(drop=True).iterrows():
        ax.hlines(j, r.pi95_lo, r.pi95_hi, color="0.75", lw=6, zorder=0,
                  label="95% profile interval" if j == 0 else None)
        for col, lab, colr, m in marks:
            ax.plot(r[col], j, m, color=colr, ms=11 if m == "|" else 7, mew=2,
                    label=lab if j == 0 else None)
    ax.set_yticks(range(len(df)), df.run)
    ax.set_xlabel("B.to_A")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("ridge (MLE -> profile mean) vs volume (profile mean -> posterior mean)")
    fig.tight_layout()
    fig.savefig(fig_dir / "decomposition.png", dpi=110)
    plt.close(fig)
    pd.DataFrame(curves).to_csv(OUT / "profile_curves.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main()
