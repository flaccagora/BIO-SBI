"""Analysis of the dedifferentiation-test study: calibration, power, mis-typing.

Reads results/exact_gw/dediff_test/{null,alt,typing,boot}/ (scripts/dediff_test/run.py) and
writes tables (CSV + markdown) to results/exact_gw/dediff_test/summary/ and figures to
docs/dediff_test/figures/.

  uv run python scripts/dediff_test/analyse.py [--root results/exact_gw/dediff_test]

Calibration methods compared at nominal 5% (per null cell and n):
  mc        Monte-Carlo critical value: 95% quantile of Lambda over the null datasets of
            that cell (oracle: needs the true nuisance); its own size is 5% by construction,
            reported is the size when the critical value comes from the *other* null cells
            (pooled), i.e. without knowing the nuisance
  boot      per-dataset parametric bootstrap (simulate from theta_0, refit): rejection
            rate on the bootstrap subset and the MC-evaluated size mean_i P_MC(Lambda >
            c*_i) of its critical values
  chibar    per-dataset chi-bar null from the observed information at theta_0
  binomial  chi-bar with weights 1/8, 3/8, 3/8, 1/8 (critical value 5.43)
  chi2_3    plain chi-square, 3 df (critical value 7.81)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from bio_sbi.inference import dediff_lrt as D  # noqa: E402

ROOT = Path("/home/matteo/BIO-SBI/results/exact_gw/dediff_test")
REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "docs" / "dediff_test" / "figures"
ALPHA = 0.05
CRIT_BIN = D.chibar_crit(D.BINOMIAL_WEIGHTS, ALPHA)
CRIT_CHI3 = float(stats.chi2.ppf(1 - ALPHA, 3))
NS = (128, 256, 512, 1024)
COLORS = {128: "#9ecae1", 256: "#4292c6", 512: "#08519c", 1024: "#08306b"}


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(c - h, 0.0), min(c + h, 1.0))


def fmt_rate(k: float, n: float) -> str:
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.0f}-{100 * hi:.0f}]"


# ---------------------------------------------------------------------- loading
def load(root: Path, study: str) -> pd.DataFrame:
    rows = []
    for f in sorted((root / study).glob("*/s*.json")):  # study includes the tag
        d = json.loads(f.read_text())
        for an, recs in d["results"].items():
            for r in recs:
                cb = r.get("chibar") or {}
                rows.append({
                    "cell": d["cell"], "base": d["base"], "route": d["route"], "m": d["m_ba"],
                    "typing": d["typing"], "analysis": an, "seed": d["seed"], "n": r["n"],
                    "lam": r["lambda"], "ll0": r["ll0"], "ll1": r["ll1"],
                    "m_hat": r["m_ba_hat"], "q_hat": r["q_hat"],
                    "p_chibar": r.get("p_chibar"), "p_binomial": r["p_binomial"],
                    "p_chi2_3": r["p_chi2_3"], "crit_chibar": cb.get("crit95"),
                    "w0": (cb.get("weights") or [np.nan] * 4)[0],
                    "w1": (cb.get("weights") or [np.nan] * 4)[1],
                    "w2": (cb.get("weights") or [np.nan] * 4)[2],
                    "w3": (cb.get("weights") or [np.nan] * 4)[3],
                    "n_bnd": cb.get("n_boundary_nuisance"),
                    "missed0": r.get("missed0", np.nan), "missed1": r.get("missed1", np.nan),
                    "h0_improved": r["h0_improved_by_full"], "tie1": r["full_starts_tied"],
                    "seconds": r["seconds"], "evals": r["evals"],
                    "P0": r["P0"], "P1": r["P1"],
                    "kkt0": (all(v <= 0 for v in r["score_dediff"].values())
                             if "score_dediff" in r else np.nan),
                })  # fmt: skip
    return pd.DataFrame(rows)


def load_boot(root: Path, tag: str = "") -> pd.DataFrame:
    rows = []
    for f in sorted((root / f"boot{tag}").glob("*/n*_s*.json")):
        d = json.loads(f.read_text())
        lam_b = np.array([b["lambda"] for b in d["boot"]])
        rows.append({"cell": d["cell"], "seed": d["seed"], "n": d["n"], "lam": d["lambda"],
                     "p_boot": d["p_boot"], "crit_boot": d["crit95_boot"], "reps": len(lam_b),
                     "p_chibar": d.get("p_chibar"),
                     "crit_chibar": (d.get("chibar") or {}).get("crit95"),
                     "lam_b": lam_b, "seconds": d["seconds"]})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- calibration
def mc_crit(lams: np.ndarray, alpha: float = ALPHA) -> float:
    return float(np.quantile(lams, 1 - alpha))


def mc_crit_ci(lams: np.ndarray, alpha: float = ALPHA, B: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    q = [np.quantile(rng.choice(lams, len(lams)), 1 - alpha) for _ in range(B)]
    return tuple(np.quantile(q, [0.05, 0.95]))


def calibration(null: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, n), g in null.groupby(["cell", "n"]):
        lam = g.lam.to_numpy()
        N = len(lam)
        c = mc_crit(lam)
        lo, hi = mc_crit_ci(lam)
        others = null[(null.n == n) & (null.cell != cell)].lam.to_numpy()
        c_other = mc_crit(others) if len(others) else np.nan
        row = {"cell": cell, "n": n, "datasets": N, "crit_mc": c, "crit_mc_lo": lo,
               "crit_mc_hi": hi, "crit_chibar_median": float(g.crit_chibar.median()),
               "crit_binomial": CRIT_BIN, "crit_chi2_3": CRIT_CHI3,
               "frac_lambda0": float(np.mean(lam < 1e-6)), "frac_kkt0": float(g.kkt0.mean()),
               "w0_mean": float(g.w0.mean()),
               "w1_mean": float(g.w1.mean()), "w2_mean": float(g.w2.mean()),
               "w3_mean": float(g.w3.mean()),
               "size_mc_other": float(np.mean(lam > c_other)), "crit_mc_other": c_other,
               "size_chibar": float(np.mean(g.p_chibar < ALPHA)),
               "size_binomial": float(np.mean(g.p_binomial < ALPHA)),
               "size_chi2_3": float(np.mean(g.p_chi2_3 < ALPHA)),
               "missed0": int((g.missed0 > 1e-3).sum()), "missed1": int((g.missed1 > 1e-3).sum()),
               "h0_improved": int((g.h0_improved > 1e-6).sum())}  # fmt: skip
        b = boot[(boot.cell == cell) & (boot.n == n)] if len(boot) else boot
        if len(b):
            row.update({
                "boot_datasets": len(b), "boot_reps": int(b.reps.median()),
                "crit_boot_median": float(b.crit_boot.median()),
                "crit_boot_q10": float(b.crit_boot.quantile(0.1)),
                "crit_boot_q90": float(b.crit_boot.quantile(0.9)),
                "size_boot_direct": float(np.mean(b.p_boot < ALPHA)),
                "size_boot_mc": float(np.mean([np.mean(lam > cb) for cb in b.crit_boot])),
                "ks_pooled_boot_vs_mc": float(stats.ks_2samp(np.concatenate(b.lam_b.to_list()),
                                                             lam).pvalue),
            })  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


def fig_null(null: pd.DataFrame, boot: pd.DataFrame, out: Path) -> None:
    cells = sorted(null.cell.unique())
    fig, axes = plt.subplots(len(cells), len(NS), figsize=(3.2 * len(NS), 2.6 * len(cells)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    x = np.linspace(0, 12, 400)
    for i, cell in enumerate(cells):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            g = null[(null.cell == cell) & (null.n == n)]
            if not len(g):
                ax.set_visible(False)
                continue
            lam = np.sort(g.lam.to_numpy())
            sf = 1 - np.arange(1, len(lam) + 1) / len(lam)
            ax.step(lam, np.maximum(sf, 1e-3), where="post", color="k", lw=1.6,
                    label=f"Monte Carlo ({len(lam)})")  # fmt: skip
            w = g[["w0", "w1", "w2", "w3"]].mean().to_numpy()
            ax.plot(x, [D.chibar_pvalue(v, w) for v in x], color="#d95f02",
                    label="chi-bar, obs. info (mean w)")  # fmt: skip
            ax.plot(x, [D.chibar_pvalue(v, D.BINOMIAL_WEIGHTS) for v in x], color="#7570b3",
                    ls="--", label="chi-bar 1/8,3/8,3/8,1/8")  # fmt: skip
            ax.plot(x, stats.chi2.sf(x, 3), color="#1b9e77", ls=":", label="chi2_3")
            b = boot[(boot.cell == cell) & (boot.n == n)] if len(boot) else boot
            if len(b):
                lb = np.sort(np.concatenate(b.lam_b.to_list()))
                ax.step(lb, np.maximum(1 - np.arange(1, len(lb) + 1) / len(lb), 1e-3),
                        where="post", color="#e7298a", lw=1,
                        label=f"bootstrap, pooled ({len(b)} datasets)")  # fmt: skip
            ax.axhline(ALPHA, color="grey", lw=0.6)
            ax.set_yscale("log")
            ax.set_ylim(5e-3, 1)
            ax.set_xlim(0, 12)
            ax.set_title(f"{cell}, n = {n}", fontsize=9)
            if i == len(cells) - 1:
                ax.set_xlabel("Lambda")
            if j == 0:
                ax.set_ylabel("P(Lambda > x)")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.suptitle("Null distribution of the LRT statistic (survival function)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_boot(cal: pd.DataFrame, boot: pd.DataFrame, out: Path) -> None:
    if not len(boot):
        return
    groups = sorted({(c, n) for c, n in zip(boot.cell, boot.n, strict=True)})
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(groups), 3.6))
    for k, (cell, n) in enumerate(groups):
        b = boot[(boot.cell == cell) & (boot.n == n)]
        jit = np.random.default_rng(k).uniform(-0.15, 0.15, len(b))
        ax.scatter(k + jit, b.crit_boot, s=10, color="#e7298a", alpha=0.7,
                   label="bootstrap (per dataset)" if k == 0 else None)  # fmt: skip
        ax.scatter(k + 0.3 + jit / 2, b.crit_chibar, s=8, color="#d95f02", alpha=0.6,
                   marker="^", label="chi-bar obs. info (per dataset)" if k == 0 else None)  # fmt: skip
        c = cal[(cal.cell == cell) & (cal.n == n)]
        if len(c):
            ax.errorbar(k - 0.3, c.crit_mc.iloc[0],
                        yerr=[[c.crit_mc.iloc[0] - c.crit_mc_lo.iloc[0]],
                              [c.crit_mc_hi.iloc[0] - c.crit_mc.iloc[0]]],
                        fmt="s", color="k", ms=5, label="Monte Carlo (90% CI)" if k == 0 else None)  # fmt: skip
    ax.axhline(CRIT_BIN, color="#7570b3", ls="--", lw=1, label="binomial weights")
    ax.axhline(CRIT_CHI3, color="#1b9e77", ls=":", lw=1, label="chi2_3")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{c}\nn={n}" for c, n in groups], fontsize=8)
    ax.set_ylabel("95% critical value of Lambda")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Critical values: Monte Carlo vs parametric bootstrap vs asymptotics", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- power
def power_table(alt: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    crit = {(c, n): mc_crit(g.lam.to_numpy()) for (c, n), g in null.groupby(["cell", "n"])}
    cmax = {n: max(v for (c, nn), v in crit.items() if nn == n) for n in null.n.unique()}
    rows = []
    for (base, route, m, n), g in alt.groupby(["base", "route", "m", "n"]):
        c = crit.get((base, n), np.nan)
        rows.append({"base": base, "route": route, "m": m, "n": n, "datasets": len(g),
                     "crit_mc": c, "power_mc": float(np.mean(g.lam > c)),
                     "power_mc_max": float(np.mean(g.lam > cmax[n])),
                     "power_chibar": float(np.mean(g.p_chibar < ALPHA)),
                     "power_binomial": float(np.mean(g.p_binomial < ALPHA)),
                     "power_chi2_3": float(np.mean(g.p_chi2_3 < ALPHA)),
                     "median_lambda": float(g.lam.median()),
                     "median_m_hat": float(g.m_hat.median()),
                     "missed1": int((g.missed1 > 1e-3).sum())})  # fmt: skip
    return pd.DataFrame(rows)


def mde(pw: pd.DataFrame, col: str = "power_mc", target: float = 0.8) -> pd.DataFrame:
    """Smallest m_BA with power >= target: linear interpolation in log m between grid
    points (None if never reached; '<' the smallest m if already reached there)."""
    rows = []
    for (base, route, n), g in pw.groupby(["base", "route", "n"]):
        g = g.sort_values("m")
        m, p = g.m.to_numpy(), g[col].to_numpy()
        val = None
        if p[0] >= target:
            val = f"<= {m[0]:g}"
        else:
            for k in range(1, len(m)):
                if p[k] >= target:
                    t = (target - p[k - 1]) / (p[k] - p[k - 1])
                    val = f"{np.exp(np.log(m[k - 1]) + t * (np.log(m[k]) - np.log(m[k - 1]))):.3f}"
                    break
            if val is None:
                val = f"> {m[-1]:g}"
        rows.append({"base": base, "route": route, "n": n, "mde80": val})
    return pd.DataFrame(rows)


def fig_power(pw: pd.DataFrame, out: Path, col: str = "power_mc", title: str = "") -> None:
    bases = sorted(pw.base.unique())
    routes = [r for r in ("toA", "divAB", "mixed") if r in set(pw.route)]
    fig, axes = plt.subplots(len(bases), len(routes), figsize=(3.4 * len(routes), 2.8 * len(bases)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    label = {"toA": "switch B->A only", "divAB": "division B->A+B only",
             "mixed": "mixed (m/2, m/4, m/8)"}  # fmt: skip
    for i, base in enumerate(bases):
        for j, route in enumerate(routes):
            ax = axes[i, j]
            for n in NS:
                g = pw[(pw.base == base) & (pw.route == route) & (pw.n == n)].sort_values("m")
                if not len(g):
                    continue
                lo, hi = zip(*[wilson(p * d, d) for p, d in zip(g[col], g.datasets, strict=True)],
                             strict=True)  # fmt: skip
                ax.plot(g.m, g[col], "o-", color=COLORS[n], label=f"n = {n}", ms=3)
                ax.fill_between(g.m, lo, hi, color=COLORS[n], alpha=0.12)
            ax.axhline(0.8, color="grey", lw=0.6, ls="--")
            ax.axhline(ALPHA, color="grey", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{base}: {label[route]}", fontsize=9)
            if i == len(bases) - 1:
                ax.set_xlabel("m_BA (A cells per B cell per generation)")
            if j == 0:
                ax.set_ylabel("power at 5%")
    axes[0, 0].legend(fontsize=7, title="clones / endpoint", title_fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- mis-typing
def typing_table(typ: pd.DataFrame, null: pd.DataFrame) -> pd.DataFrame:
    crit = {(c, n): mc_crit(g.lam.to_numpy()) for (c, n), g in null.groupby(["cell", "n"])}
    rows = []
    for (base, t, an, n), g in typ.groupby(["base", "typing", "analysis", "n"]):
        c = crit.get((base, n), np.nan)
        rows.append({"base": base, "typing": t, "analysis": an, "n": n, "datasets": len(g),
                     "fpr_mc": float(np.mean(g.lam > c)),
                     "fpr_chibar": float(np.mean(g.p_chibar < ALPHA)),
                     "fpr_chi2_3": float(np.mean(g.p_chi2_3 < ALPHA)),
                     "median_lambda": float(g.lam.median()),
                     "median_m_hat": float(g.m_hat.median())})  # fmt: skip
    return pd.DataFrame(rows)


def fig_typing(tt: pd.DataFrame, out: Path) -> None:
    bases = sorted(tt.base.unique())
    order = ("bA0.005", "bA0.01", "bA0.02", "bA0.05", "sym0.02", "sym0.05")
    typings = [t for t in order if t in set(tt.typing)]
    analyses = [a for a in ("ignored", "half", "modelled") if a in set(tt.analysis)]
    title = {"ignored": "typing errors ignored", "half": "modelled with half the true rate",
             "modelled": "known confusion matrix modelled"}  # fmt: skip
    style = {"bA0.005": ("#fdd0a2", "s"), "bA0.01": ("#fd8d3c", "s"),
             "bA0.02": ("#e6550d", "s"), "bA0.05": ("#a63603", "s"),
             "sym0.02": ("#74c476", "o"), "sym0.05": ("#238b45", "o")}  # fmt: skip
    fig, axes = plt.subplots(len(analyses), len(bases), figsize=(4.0 * len(bases),
                             2.6 * len(analyses)), sharex=True, sharey=True, squeeze=False)  # fmt: skip
    for i, an in enumerate(analyses):
        for j, base in enumerate(bases):
            ax = axes[i, j]
            for t in typings:
                g = tt[(tt.base == base) & (tt.typing == t) & (tt.analysis == an)].sort_values("n")
                if len(g):
                    ax.plot(g.n, g.fpr_mc, marker=style[t][1], color=style[t][0], ms=3,
                            label=t)  # fmt: skip
            ax.axhline(ALPHA, color="grey", lw=0.6, ls=":")
            ax.set_xscale("log", base=2)
            ax.set_xticks(NS)
            ax.set_xticklabels(NS)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{base}: {title[an]}", fontsize=9)
            if i == len(analyses) - 1:
                ax.set_xlabel("clones per endpoint")
            if j == 0:
                ax.set_ylabel("false-positive rate")
    axes[0, -1].legend(fontsize=6, loc="center right", title="data typing", title_fontsize=6)
    fig.suptitle("Mis-typing under H0 (bA = B read as A, sym = symmetric); "
                 "MC critical value of the perfectly typed null", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--tag", default="", help="folder suffix (e.g. _pilot); outputs to summary<tag>/")
    args = ap.parse_args()
    global FIG
    if args.tag:
        FIG = args.root / f"summary{args.tag}" / "figures"
    out = args.root / f"summary{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    null = load(args.root, f"null{args.tag}")
    alt = load(args.root, f"alt{args.tag}")
    typ = load(args.root, f"typing{args.tag}")
    boot = load_boot(args.root, args.tag)
    report = []
    if len(null):
        cal = calibration(null, boot)
        cal.to_csv(out / "calibration.csv", index=False)
        fig_null(null, boot, FIG / "null_distributions.png")
        fig_boot(cal, boot, FIG / "critical_values.png")
        report.append("## Calibration\n\n" + md(cal))
        diag = null.groupby("cell").agg(datasets=("seed", "nunique"), lrts=("lam", "size"),
                                        seconds=("seconds", "mean"),
                                        missed0=("missed0", lambda x: int((x > 1e-3).sum())),
                                        missed1=("missed1", lambda x: int((x > 1e-3).sum())),
                                        max_missed1=("missed1", "max"),
                                        h0_improved=("h0_improved", lambda x: int((x > 1e-6).sum())))  # fmt: skip
        report.append("## Optimisation diagnostics (null)\n\n" + md(diag.reset_index()))
    if len(alt) and len(null):
        pw = power_table(alt, null)
        pw.to_csv(out / "power.csv", index=False)
        fig_power(pw, FIG / "power_mc.png", "power_mc",
                  "Power at 5% (Monte-Carlo critical value of the base null)")  # fmt: skip
        fig_power(pw, FIG / "power_chibar.png", "power_chibar",
                  "Power at 5% (chi-bar p-value from the observed information)")  # fmt: skip
        mm = mde(pw, "power_mc").merge(mde(pw, "power_chibar"), on=["base", "route", "n"],
                                        suffixes=("_mc", "_chibar"))  # fmt: skip
        mm.to_csv(out / "mde.csv", index=False)
        report.append("## Power\n\n" + md(pw))
        report.append("## Minimum detectable m_BA at 80% power\n\n" + md(mm))
    if len(typ) and len(null):
        tt = typing_table(typ, null)
        tt.to_csv(out / "typing.csv", index=False)
        fig_typing(tt, FIG / "mistyping.png")
        report.append("## Mis-typing\n\n" + md(tt))
    (out / "tables.md").write_text("\n\n".join(report) + "\n")
    print("\n\n".join(report))


if __name__ == "__main__":
    main()
