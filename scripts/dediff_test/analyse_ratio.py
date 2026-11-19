"""Analysis of the ratio study: how precisely must the mis-typing rate be known?

Data mis-typed at the true rate e (B read as A, or symmetric), analysed with the confusion
matrix at the assumed rate r * e (scripts/dediff_test/run.py ratio). Reads
results/exact_gw/dediff_test/typing_ratio/ (plus, for reference, r = 0 = errors ignored
from typing/ and perfectly typed alternatives from alt/) and writes tables to
typing_ratio/summary/ and figures to docs/dediff_test/figures/.

  uv run python scripts/dediff_test/analyse_ratio.py

Critical values at nominal 5%:
  mc      95% Monte-Carlo quantile of Lambda over the perfectly typed null datasets of the
          same null row and n (summary/calibration.csv; as in the main study)
  chibar  per-dataset chi-bar critical value from the H0 fit of that analysis

Tolerance window (per n, m_BA, data typing, null row and route) [r_min, r_max]: r_min is
the smallest grid ratio from which on the false-positive rate is <= 7% (at r_min and every
larger grid ratio; it need not be <= 1); r_max the largest grid ratio >= r_min up to which
power stays >= 80% (variant "rel": >= the power at r = 1 minus 10 points). Empty if no
grid ratio qualifies.
Robust test: for an interval [lo, hi] of ratios, reject only if the test rejects at every
grid ratio inside it (smallest Lambda > MC critical value; largest chi-bar p < 5%).
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

ROOT = Path("/home/matteo/BIO-SBI/results/exact_gw/dediff_test")
REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "docs" / "dediff_test" / "figures"
ALPHA = 0.05
FPR_MAX = 0.07
POWER_MIN = 0.8
POWER_DROP = 0.10
NS = (128, 256, 512, 1024)
#: Robust-test intervals of the ratio r = assumed / true rate.
INTERVALS = {"exact": (1.0, 1.0), "+-10%": (0.9, 1.1), "+-25%": (0.75, 1.25),
             "+-50%": (0.5, 1.5), "x/2": (0.5, 2.0)}  # fmt: skip
# categorical slots 1-4 of the reference palette (fixed order) + marker/line encoding
SERIES = {("N1", "mixed"): ("#2a78d6", "o", "-"), ("N1", "toA"): ("#eb6834", "s", "-"),
          ("N2", "mixed"): ("#1baf7a", "o", "--"), ("N2", "toA"): ("#eda100", "s", "--")}  # fmt: skip
NULL_SERIES = {("N1", "bA0.01"): ("#2a78d6", "o", "-"), ("N1", "bA0.02"): ("#eb6834", "s", "-"),
               ("N1", "sym0.02"): ("#1baf7a", "^", "-"), ("N2", "bA0.01"): ("#2a78d6", "o", "--"),
               ("N2", "bA0.02"): ("#eb6834", "s", "--"),
               ("N2", "sym0.02"): ("#1baf7a", "^", "--")}  # fmt: skip
ROUTE_LABEL = {"mixed": "mixed", "toA": "switch only"}
KEYS = ["cell", "base", "route", "m", "typing", "seed", "n"]


def mlabel(m: float) -> str:
    """Table label of an effect size (kept as text by the markdown writer)."""
    return "H0" if m == 0 else f"m={m:g}"


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson 95% interval of a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (max(c - h, 0.0), min(c + h, 1.0))


# ---------------------------------------------------------------------- loading
def _rows(d: dict, an: str, recs: list, ratio: float) -> list[dict]:
    out = []
    for r in recs:
        cb = r.get("chibar") or {}
        out.append({"cell": d["cell"], "base": d["base"], "route": d["route"], "m": d["m_ba"],
                    "typing": d["typing"], "seed": d["seed"], "n": r["n"], "ratio": ratio,
                    "lam": r["lambda"], "p_chibar": r.get("p_chibar"),
                    "crit_chibar": cb.get("crit95"), "m_hat": r["m_ba_hat"],
                    "seconds": r["seconds"], "missed0": r.get("missed0", np.nan),
                    "missed1": r.get("missed1", np.nan),
                    "h0_improved": r["h0_improved_by_full"]})  # fmt: skip
    return out


def load(root: Path) -> pd.DataFrame:
    """One row per (dataset, n, ratio); ratio 0 = errors ignored (typing/, H0 only)."""
    rows = []
    for f in sorted((root / "typing_ratio").glob("*/s*.json")):
        d = json.loads(f.read_text())
        for an, recs in d["results"].items():
            rows += _rows(d, an, recs, float(an[1:]))
    df = pd.DataFrame(rows)
    cells = set(df.cell) if len(df) else set()
    ign = []
    for f in sorted((root / "typing").glob("*/s*.json")):
        d = json.loads(f.read_text())
        if d["cell"] in cells:
            ign += _rows(d, "ignored", d["results"]["ignored"], 0.0)
    return pd.concat([df, pd.DataFrame(ign)], ignore_index=True)


def load_perfect(root: Path, cells: set) -> pd.DataFrame:
    """Perfectly typed alternatives (alt/): the same clones without typing errors."""
    rows = []
    for f in sorted((root / "alt").glob("*/s*.json")):
        d = json.loads(f.read_text())
        if d["cell"] in cells:
            rows += _rows(d, "perfect", d["results"]["perfect"], np.nan)
    return pd.DataFrame(rows)


def mc_crits(root: Path) -> dict:
    """``{(null row, n): 95% Monte-Carlo critical value}`` of the perfectly typed null."""
    cal = pd.read_csv(root / "summary" / "calibration.csv")
    return {(r.cell, int(r.n)): float(r.crit_mc) for r in cal.itertuples()}


def add_decisions(df: pd.DataFrame, crit: dict) -> pd.DataFrame:
    """Rejection at 5% with the MC and the per-dataset chi-bar critical values."""
    df = df.copy()
    df["route"] = df.route.fillna("-")
    df["crit_mc"] = [crit[(b, int(n))] for b, n in zip(df.base, df.n, strict=True)]
    df["rej_mc"] = df.lam > df.crit_mc
    df["rej_chibar"] = df.p_chibar < ALPHA
    return df


# ---------------------------------------------------------------------- tables
def rate_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rejection rates (with Wilson intervals) per ``keys`` + n + ratio."""
    rows = []
    for k, g in df.groupby([*keys, "n", "ratio"], dropna=False):
        row = dict(zip([*keys, "n", "ratio"], k, strict=True))
        N = len(g)
        row["datasets"] = N
        for c in ("mc", "chibar"):
            x = int(g[f"rej_{c}"].sum())
            lo, hi = wilson(x, N)
            row.update({f"rate_{c}": x / N, f"rate_{c}_lo": lo, f"rate_{c}_hi": hi})
        row["median_lambda"] = float(g.lam.median())
        row["median_m_hat"] = float(g.m_hat.median())
        rows.append(row)
    return pd.DataFrame(rows)


def r_min(ratios: np.ndarray, fpr: np.ndarray, cap: float = FPR_MAX) -> float:
    """Smallest grid ratio with false-positive rate <= cap there and at all larger ratios."""
    ok = np.asarray(fpr) <= cap
    out = np.nan
    for r, k in zip(ratios[::-1], ok[::-1], strict=True):
        if not k:
            break
        out = float(r)
    return out


def r_max(ratios: np.ndarray, power: np.ndarray, lo: float, floor: float) -> float:
    """Largest grid ratio >= lo up to which power stays >= floor (nan if not at lo)."""
    out = np.nan
    if np.isnan(lo):
        return out
    for r, p in zip(ratios, power, strict=True):
        if r < lo - 1e-9:
            continue
        if p < floor:
            break
        out = float(r)
    return out


def tolerance(fpr: pd.DataFrame, pw: pd.DataFrame) -> pd.DataFrame:
    """Tolerance window per (base, route, m, typing, n) and critical value."""
    rows = []
    for (base, route, m, typing, n), g in pw.groupby(["base", "route", "m", "typing", "n"]):
        f = fpr[(fpr.base == base) & (fpr.typing == typing) & (fpr.n == n) & (fpr.ratio > 0)]
        for c in ("mc", "chibar"):
            ff = f.set_index("ratio")[f"rate_{c}"]
            pp = g.set_index("ratio")[f"rate_{c}"]
            r = np.array(sorted(set(ff.index) & set(pp.index)))
            p = pp.loc[r].to_numpy()
            p1 = float(pp.loc[1.0]) if 1.0 in pp.index else np.nan
            lo = r_min(r, ff.loc[r].to_numpy())
            rows.append({"base": base, "route": route, "m": m, "typing": typing, "n": n,
                         "crit": c, "datasets": int(g.datasets.min()),
                         "fpr_r1": float(ff.get(1.0, np.nan)), "power_r1": p1,
                         "power_r2": float(pp.get(2.0, np.nan)), "r_min": lo,
                         "r_max80": r_max(r, p, lo, POWER_MIN),
                         "r_max_rel": r_max(r, p, lo, p1 - POWER_DROP)})  # fmt: skip
    return pd.DataFrame(rows)


def robust(df: pd.DataFrame) -> pd.DataFrame:
    """Robust test per dataset and interval: reject iff every grid ratio inside rejects."""
    d = df[df.ratio > 0]
    rows = []
    for name, (lo, hi) in INTERVALS.items():
        s = d[(d.ratio >= lo - 1e-9) & (d.ratio <= hi + 1e-9)]
        g = s.groupby(KEYS)
        agg = g.agg(lam_min=("lam", "min"), p_max=("p_chibar", "max"),
                    rej_mc=("rej_mc", "all"), rej_chibar=("rej_chibar", "all"),
                    k=("ratio", "size"))  # fmt: skip
        # decision of the plain test at the upper end of the interval, for comparison
        top = s[np.isclose(s.ratio, hi)].set_index(KEYS)[["rej_mc", "rej_chibar"]]
        agg = agg.join(top.add_suffix("_hi"))
        # complete datasets only (every grid ratio of the interval analysed)
        need = len(set(np.round(s.ratio, 6)))
        agg = agg[agg.k == need].reset_index()
        agg["interval"] = name
        agg["lo"], agg["hi"] = lo, hi
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def robust_table(rb: pd.DataFrame) -> pd.DataFrame:
    """Robust-test rejection rates per interval, cell and n."""
    rows = []
    for k, g in rb.groupby(["interval", "base", "route", "m", "typing", "n"]):
        row = dict(zip(["interval", "base", "route", "m", "typing", "n"], k, strict=True))
        row.update({"datasets": len(g), "grid_points": int(g.k.median()),
                    "rate_mc": float(g.rej_mc.mean()), "rate_chibar": float(g.rej_chibar.mean()),
                    "same_as_hi_mc": float(np.mean(g.rej_mc == g.rej_mc_hi)),
                    "same_as_hi_chibar": float(np.mean(g.rej_chibar == g.rej_chibar_hi))})  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- figures
def _ratio_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ticks = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvline(1.0, color="#bbbbbb", lw=0.6)
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_fpr(ft: pd.DataFrame, out: Path) -> None:
    """False-positive rate vs ratio; columns n, rows critical value."""
    fig, axes = plt.subplots(2, len(NS), figsize=(3.1 * len(NS), 5.4), sharex=True, sharey=True)
    ft = ft[ft.ratio > 0]
    for i, c in enumerate(("mc", "chibar")):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            for (base, typ), (col, mk, ls) in NULL_SERIES.items():
                g = ft[(ft.base == base) & (ft.typing == typ) & (ft.n == n)].sort_values("ratio")
                if len(g):
                    ax.plot(g.ratio, g[f"rate_{c}"], color=col, marker=mk, ls=ls, lw=1.6, ms=4,
                            label=f"{base}, {typ}")  # fmt: skip
            ax.axhline(ALPHA, color="#555555", lw=0.8, ls=":")
            ax.axhline(FPR_MAX, color="#999999", lw=0.6, ls="--")
            _ratio_axis(ax)
            ax.set_ylim(-0.01, 1.02)
            if i == 0:
                ax.set_title(f"n = {n} clones / endpoint", fontsize=9)
            if i == 1:
                ax.set_xlabel("assumed / true error rate r")
            if j == 0:
                ax.set_ylabel(f"false-positive rate\n({'MC' if c == 'mc' else 'chi-bar'} critical"
                              " value)")  # fmt: skip
    axes[0, -1].legend(fontsize=6.5, title="null row, data typing", title_fontsize=6.5)
    fig.suptitle("H0 true, data mis-typed (bA = KRT20+ read as LGR5+, sym = both ways): "
                 "false positives at 5% vs assumed error rate (dotted 5%, dashed 7%)",
                 fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_power(pt: pd.DataFrame, typing: str, out: Path, crit: str = "mc") -> None:
    """Power vs ratio; rows m_BA, columns n; lines null row x route."""
    ms = sorted(pt.m.unique())
    fig, axes = plt.subplots(len(ms), len(NS), figsize=(3.1 * len(NS), 2.3 * len(ms)),
                             sharex=True, sharey=True, squeeze=False)  # fmt: skip
    for i, m in enumerate(ms):
        for j, n in enumerate(NS):
            ax = axes[i, j]
            for (base, route), (col, mk, ls) in SERIES.items():
                g = pt[(pt.base == base) & (pt.route == route) & np.isclose(pt.m, m) & (pt.n == n)
                       & (pt.typing == typing)].sort_values("ratio")  # fmt: skip
                if len(g):
                    ax.plot(g.ratio, g[f"rate_{crit}"], color=col, marker=mk, ls=ls, lw=1.6,
                            ms=4, label=f"{base}, {ROUTE_LABEL[route]}")  # fmt: skip
            ax.axhline(POWER_MIN, color="#999999", lw=0.6, ls="--")
            _ratio_axis(ax)
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(f"n = {n}", fontsize=9)
            if i == len(ms) - 1:
                ax.set_xlabel("assumed / true error rate r")
            if j == 0:
                ax.set_ylabel(f"m_BA = {m:g}\npower")
    axes[0, 0].legend(fontsize=6.5, title="null row, route", title_fontsize=6.5)
    fig.suptitle(f"Power at 5% vs assumed error rate; data mis-typed {typing} "
                 f"({'MC' if crit == 'mc' else 'chi-bar'} critical value; dashed 80%)",
                 fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_robust(rt: pd.DataFrame, out: Path, crit: str = "mc") -> None:
    """Robust test: false-positive rate (H0) and power (H1) vs interval width."""
    names = list(INTERVALS)
    x = np.arange(len(names))
    typings = [t for t in ("bA0.01", "bA0.02") if t in set(rt.typing)]
    fig, axes = plt.subplots(len(typings), 1 + len(NS), figsize=(3.0 * (1 + len(NS)),
                             2.7 * len(typings)), squeeze=False)  # fmt: skip
    for i, typ in enumerate(typings):
        ax = axes[i, 0]
        for (base, t2), (col, mk, ls) in NULL_SERIES.items():
            if t2 != typ:
                continue
            for n, a in zip(NS, (0.35, 0.55, 0.75, 1.0), strict=True):
                g = rt[(rt.m == 0) & (rt.base == base) & (rt.typing == typ) & (rt.n == n)]
                g = g.set_index("interval").reindex(names)
                ax.plot(x, g[f"rate_{crit}"], color=col, marker=mk, ls=ls, alpha=a, lw=1.2,
                        ms=3, label=f"{base}, n={n}")  # fmt: skip
        ax.axhline(ALPHA, color="#555555", lw=0.8, ls=":")
        ax.set_ylim(-0.005, 0.2)
        ax.set_title(f"H0, data {typ}: false positives", fontsize=9)
        ax.legend(fontsize=5.5, ncol=2)
        for j, n in enumerate(NS):
            ax = axes[i, 1 + j]
            for m, a in zip((0.005, 0.01, 0.05, 0.1), (0.4, 0.6, 0.8, 1.0), strict=True):
                for (base, route), (col, mk, ls) in SERIES.items():
                    g = rt[np.isclose(rt.m, m) & (rt.base == base) & (rt.route == route) &
                           (rt.typing == typ) & (rt.n == n)]  # fmt: skip
                    if not len(g):
                        continue
                    g = g.set_index("interval").reindex(names)
                    ax.plot(x, g[f"rate_{crit}"], color=col, marker=mk, ls=ls, alpha=a, lw=1.2,
                            ms=3, label=f"{base} {ROUTE_LABEL[route]}" if m == 0.1 else None)  # fmt: skip
            ax.axhline(POWER_MIN, color="#999999", lw=0.6, ls="--")
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"H1, data {typ}: power, n = {n}\n(faint m=0.005 ... dark m=0.1)",
                         fontsize=8)  # fmt: skip
            if j == 0:
                ax.legend(fontsize=5.5)
        for ax in axes[i]:
            ax.set_xticks(x)
            ax.set_xticklabels(names, fontsize=7)
            ax.grid(axis="y", color="#eeeeee", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle("Robust test (reject only if the test rejects at every assumed rate in the "
                 f"interval around the true rate; {'MC' if crit == 'mc' else 'chi-bar'} "
                 "critical value)", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Markdown table."""
    return df.to_markdown(index=False, floatfmt=floatfmt)


def pivot_pct(df: pd.DataFrame, index: list[str], col: str) -> pd.DataFrame:
    """Percent table with ratios as columns."""
    if "m" in df:
        df = df.assign(m=df.m.map(mlabel))
    t = df.pivot_table(index=index, columns="ratio", values=col).mul(100).round(0)
    t.columns = [f"r={c:g}" for c in t.columns]
    return t.reset_index()


def main() -> None:
    """Build tables and figures."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    out = args.root / "typing_ratio" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    crit = mc_crits(args.root)
    df = add_decisions(load(args.root), crit)
    df.drop(columns=[]).to_csv(out / "per_dataset.csv", index=False)
    h0, h1 = df[df.m == 0], df[df.m > 0]
    report = [f"# Ratio study (assumed / true mis-typing rate)\n\nanalyses: {len(df)} "
              f"(H0 {len(h0)}, H1 {len(h1)}); datasets per cell: H0 "
              f"{h0.groupby(['cell', 'n', 'ratio']).size().describe()[['min', 'max']].tolist()}"
              f", H1 {h1.groupby(['cell', 'n', 'ratio']).size().describe()[['min', 'max']].tolist()}"
              ]  # fmt: skip
    ft = rate_table(h0, ["base", "typing"])
    ft.to_csv(out / "fpr.csv", index=False)
    fig_fpr(ft, FIG / "ratio_fpr.png")
    for c in ("mc", "chibar"):
        report.append(f"## False-positive rate (%), {c} critical value\n\n"
                      + md(pivot_pct(ft, ["base", "typing", "n"], f"rate_{c}"), ".0f"))  # fmt: skip
    if len(h1):
        pt = rate_table(h1, ["base", "route", "m", "typing"])
        perf = add_decisions(load_perfect(args.root, {c.rsplit("_", 1)[0] for c in h1.cell}),
                             crit)  # fmt: skip
        if len(perf):
            seeds = h1.groupby("cell").seed.max().to_dict()
            perf = perf[[s <= seeds.get(f"{c}_bA0.02", -1) for c, s in
                         zip(perf.cell, perf.seed, strict=True)]]  # fmt: skip
            pp = rate_table(perf, ["base", "route", "m"])
            pp.to_csv(out / "power_perfect.csv", index=False)
        pt.to_csv(out / "power.csv", index=False)
        for typ in sorted(h1.typing.unique()):
            fig_power(pt, typ, FIG / f"ratio_power_{typ}.png", "mc")
            fig_power(pt, typ, FIG / f"ratio_power_{typ}_chibar.png", "chibar")
        for c in ("mc", "chibar"):
            report.append(f"## Power (%), {c} critical value\n\n"
                          + md(pivot_pct(pt, ["typing", "m", "base", "route", "n"],
                                         f"rate_{c}"), ".0f"))  # fmt: skip
        if len(perf):
            report.append("## Power (%) with perfect typing (same clones)\n\n"
                          + md(pp.assign(m=pp.m.map(mlabel)).pivot_table(index=["m", "base", "route"], columns="n",
                                              values="rate_mc").mul(100).round(0).reset_index(),
                               ".0f"))  # fmt: skip
        tol = tolerance(ft, pt)
        tol.to_csv(out / "tolerance.csv", index=False)
        tol = tol.assign(m=tol.m.map(mlabel))
        report.append("## Tolerance windows [r_min, r_max] (FPR <= 7% for all r >= r_min; "
                      "power >= 80% [r_max80] or >= power(r=1) - 10 points [r_max_rel] for "
                      "r_min <= r <= r_max)\n\n" + md(tol))
        worst = tol.groupby(["crit", "typing", "m", "n"]).agg(
            r_min=("r_min", lambda x: x.max() if x.notna().all() else np.nan),
            r_max80=("r_max80", lambda x: x.min() if x.notna().all() else np.nan),
            r_max_rel=("r_max_rel", lambda x: x.min() if x.notna().all() else np.nan),
            power_r1_min=("power_r1", "min"), power_r2_min=("power_r2", "min")).reset_index()
        for k in ("r_max80", "r_max_rel"):  # intersection over rows: empty if lo > hi
            worst[k] = worst[k].where(worst[k] >= worst.r_min)
        worst["window80"] = [f"[{a:g}, {b:g}]" if np.isfinite(b) else "empty"
                             for a, b in zip(worst.r_min, worst.r_max80, strict=True)]  # fmt: skip
        worst.to_csv(out / "tolerance_worst.csv", index=False)
        report.append("## Tolerance windows, most restrictive over null rows N1/N2 and routes"
                      "\n\n" + md(worst, ".2f"))
    rb = robust(df)
    rt = robust_table(rb)
    rt.to_csv(out / "robust.csv", index=False)
    for c in ("mc", "chibar"):
        t = rt.assign(m=rt.m.map(mlabel)).pivot_table(
            index=["typing", "m", "base", "route", "n"], columns="interval",
            values=f"rate_{c}").mul(100).round(0)  # fmt: skip
        t = t[[k for k in INTERVALS if k in t.columns]].reset_index()
        report.append(f"## Robust test: rejection rate (%), {c} critical value\n\n"
                      + md(t, ".0f"))  # fmt: skip
        fig_robust(rt, FIG / f"ratio_robust{'' if c == 'mc' else '_chibar'}.png", c)
    report.append("## Robust test: fraction of datasets where it decides like the plain test "
                  "at the upper end of the interval\n\n"
                  + md(rt.groupby(["interval"]).agg(mc=("same_as_hi_mc", "mean"),
                                                    chibar=("same_as_hi_chibar", "mean"))
                       .reset_index()))  # fmt: skip
    r1 = df[df.ratio == 1.0]
    report.append(f"## Optimisation diagnostics\n\nr = 1 fits ({len(r1)} LRTs) where a fit "
                  f"started at the truth beat the multistart by > 1e-3 nats: H0 "
                  f"{int((r1.missed0 > 1e-3).sum())}, full {int((r1.missed1 > 1e-3).sum())} "
                  f"(max {r1.missed0.max():.2g} / {r1.missed1.max():.2g}); H0 fit improved by a "
                  f"full-model start in {int((df.h0_improved > 1e-6).sum())} of {len(df)} LRTs;"
                  f" mean seconds per LRT {df.seconds.mean():.1f}")
    (out / "tables.md").write_text("\n\n".join(report) + "\n")
    print("\n\n".join(report))


if __name__ == "__main__":
    main()
