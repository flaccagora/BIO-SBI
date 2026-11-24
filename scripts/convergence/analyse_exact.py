"""Tables and figures of the composition-convergence study, parts 1-2 (exact law + prior map).

Reads ``OUT/exact/*.json`` (:mod:`compute_exact`) and writes to ``docs/convergence``:

* ``tables/by_generation.csv``: per set and h, the exact composition summaries + theory;
* ``tables/truths.{csv,md}``: per XG2 truth / study set: spectrum, regime, N_mean(0.01,
  0.001; exact and theory), N_clone(0.1, 0.05), fitted per-generation sd ratio vs theory;
* ``tables/prior_draws.csv`` (2000 plausible flat-prior draws), ``tables/prior_summary.md``,
  ``tables/no_dediff.{csv,md}`` (c = 0 limit);
* ``figures/truths_composition.png`` (pooled / per-clone composition vs h per truth),
  ``figures/gap_sd_theory.png`` (|pooled - u_A| and per-clone sd vs h with theory lines),
  ``figures/prior_map.png`` (distribution of N; N_mean over (coupling, a - d) at rho = 1.5).

  python scripts/convergence/analyse_exact.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import (  # noqa: E402
    FIG,
    OUT,
    SHORTNAME,
    TAB,
    TRUTHS,
    gap_amplitude,
    mean_matrix,
    n_mean,
    n_mean_theory,
    pooled_fraction,
    prior_draws,
    spectrum,
    truth_xg2,  # fmt: skip
)
from sets import SETS  # noqa: E402

EXACT = OUT / "exact"
HALO = [pe.withStroke(linewidth=2.5, foreground="w")]
H_FIT = 6  # sd fits use h >= H_FIT (past the all-A / tiny-clone transient)


def load(label: str) -> pd.DataFrame | None:
    f = EXACT / f"{label}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    df = pd.DataFrame(d["rows"])
    df["label"] = label
    return df


def theory_cols(df: pd.DataFrame, M: np.ndarray) -> pd.DataFrame:
    s = spectrum(M)
    C, D = gap_amplitude(M)
    h = df["h"].to_numpy()
    sg = np.sign(s["lambda2"]) ** h
    df["u_A"] = s["u_A"]
    df["pooled_theory"] = s["u_A"] + C * sg * s["r"] ** h / (1 + D * sg * s["r"] ** h)
    df["gap"] = np.abs(df["pooled"] - s["u_A"])
    df["gap_mean_frac"] = np.abs(df["mean_frac"] - s["u_A"])
    df["gap_lead"] = abs(C) * s["r"] ** h
    return df


def n_clone(df: pd.DataFrame, delta: float, rate: float) -> tuple[float, bool]:
    """First h with sd < delta at every computed later h; extrapolated with the fitted
    per-generation ratio beyond the last computed h (flag True).
    """
    h, sd = df["h"].to_numpy(), df["sd_frac"].to_numpy()
    bad = np.flatnonzero(sd >= delta)
    if len(bad) == 0:
        return float(h[0]), False
    if bad[-1] < len(h) - 1:
        return float(h[bad[-1] + 1]), False
    if not np.isfinite(rate) or rate <= 1:
        return np.inf, True
    return float(h[-1] + np.ceil(np.log(sd[-1] / delta) / np.log(rate))), True


def sd_rate(df: pd.DataFrame) -> float:
    """Fitted per-generation shrink factor of the per-clone sd (h >= H_FIT, last 6 h)."""
    d = df[df["h"] >= H_FIT].tail(6)
    if len(d) < 3:
        return np.nan
    return float(np.exp(-np.polyfit(d["h"], np.log(d["sd_frac"]), 1)[0]))


def summary_row(label: str, p12: np.ndarray, df: pd.DataFrame | None) -> dict:
    M = mean_matrix(p12)
    s = spectrum(M)
    C, _ = gap_amplitude(M)
    # per-generation shrink factor of the per-clone sd: sd ~ max(rho^(-h/2), r^h)
    rate_th = min(np.sqrt(s["rho"]), 1 / s["r"] if s["r"] > 0 else np.inf)
    row = {"set": label, **{k: s[k] for k in ("a", "b", "c", "d", "rho", "lambda2", "r", "u_A",
                                              "coupling", "a_minus_d")},
           "lambda2_sq_over_rho": s["lambda2"] ** 2 / s["rho"],
           "regime": "lambda2^2<rho" if s["lambda2"] ** 2 < s["rho"] else "lambda2^2>rho",
           "C": C, "N_mean_01": n_mean(M, 0.01), "N_mean_001": n_mean(M, 0.001),
           "N_mean_01_theory": n_mean_theory(M, 0.01),
           "N_mean_001_theory": n_mean_theory(M, 0.001),
           "sd_rate_theory": rate_th}  # fmt: skip
    if df is not None and len(df):
        rate = sd_rate(df)
        row["sd_rate_fit"] = rate
        row["h_max"] = int(df["h"].max())
        for dl, key in ((0.1, "N_clone_10"), (0.05, "N_clone_05")):
            n, ext = n_clone(df, dl, rate)
            row[key], row[key + "_extrap"] = n, ext
        for hh in (4, 10):
            r = df[df["h"] == hh]
            if len(r):
                row[f"sd_h{hh}"] = float(r["sd_frac"].iloc[0])
                row[f"cells_h{hh}"] = float(r["mean_cells"].iloc[0])
    return row


# ---------------------------------------------------------------------- truths
def truths_part() -> pd.DataFrame:
    sets = {gt: truth_xg2(gt) for gt in TRUTHS} | dict(SETS)
    long, rows = [], []
    for lab, p in sets.items():
        df = load(lab)
        if df is not None:
            long.append(theory_cols(df, mean_matrix(p)))
        rows.append(summary_row(lab, p, df))
    long = pd.concat(long)
    long.to_csv(TAB / "by_generation.csv", index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(TAB / "truths.csv", index=False)
    show = tab.assign(set=tab["set"].map(lambda x: SHORTNAME.get(x, x)))
    cols = ["set", "rho", "lambda2", "r", "lambda2_sq_over_rho", "u_A", "coupling", "a_minus_d",
            "N_mean_01", "N_mean_01_theory", "N_mean_001", "N_clone_10", "N_clone_05",
            "sd_rate_fit", "sd_rate_theory", "sd_h4", "sd_h10", "cells_h10"]  # fmt: skip
    (TAB / "truths.md").write_text(show[cols].round(3).to_markdown(index=False) + "\n")
    print(show[cols].round(3).to_string())
    fig_truths(long, sets)
    fig_gap_sd(long, sets)
    return tab


def fig_truths(long: pd.DataFrame, sets: dict) -> None:
    labs = TRUTHS + ["S0", "S1", "S3", "F1"]
    fig, axs = plt.subplots(3, 5, figsize=(16, 9.5), sharex=True, sharey=True)
    for ax, lab in zip(axs.flat, labs, strict=False):
        d = long[long["label"] == lab]
        M = mean_matrix(sets[lab])
        s = spectrum(M)
        hh = np.arange(0, 16)
        ax.fill_between(d["h"], d["q10"], d["q90"], color="C0", alpha=0.2, lw=0,
                        label="per clone 10-90%")  # fmt: skip
        ax.plot(d["h"], d["mean_frac"] + d["sd_frac"], color="C0", lw=0.8, ls=":")
        ax.plot(d["h"], d["mean_frac"] - d["sd_frac"], color="C0", lw=0.8, ls=":",
                label="mean ± sd")  # fmt: skip
        ax.plot(d["h"], d["mean_frac"], "o-", color="C0", ms=3, label="mean per-clone fraction")
        ax.plot(hh, pooled_fraction(M, hh), "-", color="C3", lw=1.5, label="pooled (theory)")
        ax.plot(d["h"], d["pooled"], "s", color="C3", ms=3, mfc="none", label="pooled (exact)")
        ax.axhline(s["u_A"], color="k", lw=0.8, ls="--", label="u_A")
        nm = n_mean(M, 0.01)
        ax.axvline(nm, color="C3", lw=0.8, alpha=0.6)
        ax.set_title(f"{SHORTNAME.get(lab, lab)}: ρ={s['rho']:.2f} λ₂={s['lambda2']:+.2f} "
                     f"N₀.₀₁={nm:.0f}", fontsize=9)  # fmt: skip
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    for ax in axs.flat[len(labs) :]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=7, loc="lower left")
    for ax in axs[-1]:
        ax.set_xlabel("generation h")
    for ax in axs[:, 0]:
        ax.set_ylabel("A fraction (detected A-founded clones)")
    fig.suptitle("A/B composition of A-founded clones vs generation (exact law, detected clones); "
                 "red line: generation from which the pooled fraction stays within 0.01 of u_A",
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "truths_composition.png", dpi=150)
    plt.close(fig)


def fig_gap_sd(long: pd.DataFrame, sets: dict) -> None:
    labs = TRUTHS + ["S0", "S1", "S3", "F1"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(labs):
        d = long[long["label"] == lab]
        s = spectrum(mean_matrix(sets[lab]))
        c = cmap(i)
        name = SHORTNAME.get(lab, lab)
        ok = d["gap"] > 1e-12
        axs[0].semilogy(d["h"][ok], d["gap"][ok], "o", color=c, ms=3.5)
        axs[0].semilogy(d["h"], d["gap_lead"], "-", color=c, lw=1, label=f"{name} r={s['r']:.2f}")
        axs[1].semilogy(d["h"], d["sd_frac"], "o-", color=c, ms=3, lw=0.8, label=name)
        # theory: anchored at h = 10 (or last), slope max(rho^-1/2, r)
        rate = max(1 / np.sqrt(s["rho"]), s["r"])
        h0 = min(10, int(d["h"].max()))
        sd0 = float(d.loc[d["h"] == h0, "sd_frac"].iloc[0])
        hh = np.arange(4, 16)
        axs[1].semilogy(hh, sd0 * rate ** (hh - h0), "--", color=c, lw=0.8)
        axs[2].plot(d["h"], d["sd_frac"] * np.maximum(rate, 1e-9) ** (-d["h"]), "o-", color=c,
                    ms=3, lw=0.8)  # fmt: skip
    axs[0].axhline(0.01, color="k", lw=0.8, ls=":")
    axs[0].axhline(0.001, color="k", lw=0.8, ls=":")
    axs[0].set(xlabel="generation h", ylabel="|pooled A fraction − u_A|", ylim=(1e-8, 1),
               title="pooled composition: exact (dots), |C| rᴴ (lines)")  # fmt: skip
    axs[1].axhline(0.1, color="k", lw=0.8, ls=":")
    axs[1].axhline(0.05, color="k", lw=0.8, ls=":")
    axs[1].set(xlabel="generation h", ylabel="sd of per-clone A fraction",
               title="per-clone composition: exact (dots), slope max(ρ^-½, r) (dashed)")  # fmt: skip
    axs[2].set_yscale("log")
    axs[2].set(xlabel="generation h", ylabel="sd / max(ρ^-½, r)ᴴ",
               title="sd rescaled by the theory rate (flat = theory)")  # fmt: skip
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].legend(fontsize=7, ncol=2)
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "gap_sd_theory.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- prior map
def no_dediff(p12: np.ndarray) -> np.ndarray:
    """C = 0 limit: B->A, B->AA, B->AB set to 0, the rest of B's row rescaled."""
    q = np.asarray(p12, float).copy()
    z = [7, 9, 10]
    rest = [6, 8, 11]
    q[rest] = q[rest] / q[rest].sum()
    q[z] = 0.0
    return q


def prior_part() -> pd.DataFrame:
    draws, tried = prior_draws(2000)
    rows = []
    for i, p in enumerate(draws):
        lab = f"prior{i:04d}"
        r = summary_row(lab, p, load(lab))
        r["i"] = i
        q = no_dediff(p)
        Mq = mean_matrix(q)
        sq = spectrum(Mq)
        r["c0_rho"], r["c0_u_A"], r["c0_r"] = sq["rho"], sq["u_A"], sq["r"]
        r["c0_N_mean_01"] = n_mean(Mq, 0.01, hmax=5000)
        r["disc_over_rho"] = (r["rho"] - r["lambda2"]) / r["rho"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "prior_draws.csv", index=False)
    ex = df[df["N_clone_05"].notna()]
    lines = [f"# Plausible flat-prior draws (n = {len(df)}, accepted of {tried} drawn: 30-300 "
             "expected cells at generation 10, A founder)", "",
             f"Exact per-clone law available for {len(ex)} draws.", ""]  # fmt: skip
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    cols = ["rho", "lambda2", "r", "lambda2_sq_over_rho", "coupling", "a_minus_d", "u_A",
            "N_mean_01", "N_mean_001", "N_clone_10", "N_clone_05", "sd_rate_fit",
            "sd_rate_theory"]  # fmt: skip
    qt = df[cols].quantile(q).T
    qt["max"] = df[cols].max()
    lines += ["## Quantiles", "", qt.round(3).to_markdown(), ""]
    shares = []
    for col in ("N_mean_01", "N_mean_001", "N_clone_10", "N_clone_05"):
        d = df[col].dropna()
        shares.append({"quantity": col, "n": len(d), **{f"> {k}": float((d > k).mean())
                                                         for k in (4, 10, 20)}})  # fmt: skip
    lines += ["## Share of plausible parameter sets with N above 4 / 10 / 20 generations", "",
              pd.DataFrame(shares).round(4).to_markdown(index=False), ""]  # fmt: skip
    reg = (df["lambda2"] ** 2 < df["rho"]).mean()
    lines += [f"Regime: lambda2^2 < rho (CLT fluctuations rho^(-h/2)) in {reg:.1%} of draws.",
              ""]  # fmt: skip
    # drivers: Spearman correlations and a regression of log N on the descriptors
    from scipy.stats import spearmanr

    drv = []
    for x in ("coupling", "a_minus_d", "disc_over_rho", "r", "rho", "u_A"):
        xv = df[x].abs() if x == "a_minus_d" else df[x]
        for y in ("N_mean_01", "N_clone_05"):
            d = pd.concat([xv, df[y]], axis=1).dropna()
            drv.append({"x": "|a-d|" if x == "a_minus_d" else x, "y": y,
                        "spearman": spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]})  # fmt: skip
    lines += ["## What drives N (Spearman rank correlation)", "",
              pd.DataFrame(drv).pivot(index="x", columns="y", values="spearman").round(3)
              .to_markdown(), ""]  # fmt: skip
    # c = 0 limit
    c0 = df[["c0_rho", "c0_u_A", "c0_r", "c0_N_mean_01"]]
    lines += ["## No-dedifferentiation limit c = 0 (B->A, B->AA, B->AB set to 0, B row "
              "rescaled)", "", c0.quantile(q).T.round(3).to_markdown(), "",
              f"u_A = 0 (B outgrows A: composition drifts to all B): "
              f"{(df['c0_u_A'] == 0).mean():.1%}; N_mean > 10: "
              f"{(df['c0_N_mean_01'] > 10).mean():.1%}, > 20: "
              f"{(df['c0_N_mean_01'] > 20).mean():.1%}.", ""]  # fmt: skip
    (TAB / "prior_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    fig_prior(df)
    return df


def map_grid(rho: float = 1.5, n: int = 161):
    """N_mean(0.01) over (coupling b c, a - d) at fixed rho, with b = c = sqrt(bc)."""
    kap = np.logspace(-4, np.log10(2.5), n)
    dd = np.linspace(-1.5, 1.5, n)
    N = np.full((n, n), np.nan)
    R = np.full((n, n), np.nan)
    for i, de in enumerate(dd):
        for j, k in enumerate(kap):
            disc = np.sqrt(de**2 + 4 * k)
            tr = 2 * rho - disc
            a, d = (tr + de) / 2, (tr - de) / 2
            b = c = np.sqrt(k)
            if a < 0 or d < 0 or a + b > 2 or c + d > 2:
                continue
            M = np.array([[a, b], [c, d]])
            R[i, j] = abs(rho - disc) / rho
            N[i, j] = n_mean(M, 0.01, hmax=400)
    return kap, dd, N, R


def fig_prior(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axs[0]
    bins = np.arange(0.5, 31.5, 1)
    ax.hist(df["N_mean_01"], bins=bins, alpha=0.6, label="N_mean (pooled within 0.01)")
    ax.hist(df["N_mean_001"], bins=bins, alpha=0.5, label="N_mean (within 0.001)")
    ex = df.dropna(subset=["N_clone_05"])
    ax.hist(ex["N_clone_10"].clip(upper=30), bins=bins, histtype="step", lw=2, color="C2",
            label=f"N_clone (sd < 0.10), n={len(ex)}")  # fmt: skip
    ax.hist(
        ex["N_clone_05"].clip(upper=30),
        bins=bins,
        histtype="step",
        lw=2,
        color="C3",
        label="N_clone (sd < 0.05)",
    )
    ax.set(xlabel="generations", ylabel="plausible flat-prior draws",
           title=f"{len(df)} plausible prior draws (30-300 cells at h=10)")  # fmt: skip
    ax.legend(fontsize=8)
    ax = axs[1]
    sc = ax.scatter(df["disc_over_rho"], df["N_mean_01"], c=np.log10(df["coupling"]), s=6,
                    cmap="viridis")  # fmt: skip
    x = np.linspace(0.02, 2, 200)
    r = np.abs(1 - x)
    for C, ls in ((0.3, ":"), (0.6, "--")):
        ax.plot(x, np.maximum(np.log(C / 0.01) / np.log(1 / r), 0), ls, color="k", lw=0.8,
                label=f"log(C/0.01)/log(1/r), C={C}")  # fmt: skip
    ax.set_yscale("log")
    ax.set(xlabel="spectral gap (ρ − λ₂)/ρ = √((a−d)² + 4bc)/ρ", ylabel="N_mean (0.01)",
           ylim=(0.8, 300), title="pooled convergence vs spectral gap")  # fmt: skip
    plt.colorbar(sc, ax=ax, label="log10 coupling b·c")
    ax.legend(fontsize=8)
    ax = axs[2]
    kap, dd, N, R = map_grid()
    lev = [1, 2, 4, 6, 10, 20, 40, 100, 400]
    cf = ax.contourf(kap, dd, np.log10(np.clip(N, 1, 400)), levels=np.log10(lev),
                     cmap="magma_r")  # fmt: skip
    cs = ax.contour(kap, dd, N, levels=[4, 10, 20, 40], colors="w", linewidths=0.8)
    ax.clabel(cs, fmt="%d", fontsize=8)
    band = df[(df["rho"] > 1.45) & (df["rho"] < 1.55)]
    ax.scatter(band["coupling"], band["a_minus_d"], s=4, color="c", alpha=0.6,
               label=f"prior draws, ρ∈(1.45,1.55) (n={len(band)})")  # fmt: skip
    for lab, p in SETS.items():
        s = spectrum(mean_matrix(p))
        ax.plot(s["coupling"], s["a_minus_d"], "*", ms=11, color="lime", mec="k")
        ax.annotate(lab, (s["coupling"], s["a_minus_d"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color="k", path_effects=HALO)  # fmt: skip
    for gt in TRUTHS:
        s = spectrum(mean_matrix(truth_xg2(gt)))
        ax.plot(s["coupling"], s["a_minus_d"], "^", ms=6, color="w", mec="k")
        ax.annotate(SHORTNAME[gt], (s["coupling"], s["a_minus_d"]), xytext=(3, -9),
                    textcoords="offset points", fontsize=7, color="k", path_effects=HALO)  # fmt: skip
    ax.set_xscale("log")
    ax.set(xlabel="coupling b·c (B daughters of A × A daughters of B)", ylabel="a − d",
           title="N_mean(0.01) at ρ = 1.5 (b = c); white: infeasible")  # fmt: skip
    cb = plt.colorbar(cf, ax=ax, label="N_mean (generations)")
    cb.set_ticks(np.log10(lev))
    cb.set_ticklabels([str(v) for v in lev])
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "prior_map.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    truths_part()
    prior_part()


if __name__ == "__main__":
    main()
