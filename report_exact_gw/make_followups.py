"""Figures and tables of the follow-up section of the exact-GW report.

Usage (from the repository root)::

    uv run python report_exact_gw/make_followups.py [--results /path/to/results]

Reads, under ``<results>/exact_gw/``: ``data_size/`` (XG2 incl. n = 2048/4096 for T1-T4,
with ``mle.json``), ``data_size_flat_btoa/`` (XG2 repeated with the flat-B->A prior and the
ESS/R-hat stopping rule), ``data_size_reruns/bimodal/`` (8-chain reruns of the R-hat > 1.05
cells), ``data_size_reruns/sampler_noise/`` (dataset 0, sampler seeds 1, 2) and
``reparam_bench/runs.csv``. Writes into ``report_exact_gw/figures`` and ``tables``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures as mf  # noqa: E402  (house style, helpers)

plt = mf.plt
BLUE, ORANGE, AQUA, INK, INK2 = mf.BLUE, mf.ORANGE, mf.AQUA, mf.INK, mf.INK2
GTS, LABEL, PARAMS = mf.GTS, mf.LABEL, mf.PARAMS
CUR = ["T1_ref", "T2_toA05", "T4_slowA", "T3_Bdiv"]


def load_root(root: Path) -> pd.DataFrame:
    """One row per (run, parameter) from recovery.csv + run diagnostics."""
    rows = []
    for m in sorted(root.glob("*/n*_s*/metrics.json")):
        run = m.parent
        gt = run.parent.name
        n, s = run.name[1:].split("_s")
        met = json.loads(m.read_text())
        e = met["exact_gw"]
        stop = e.get("stop") or {}
        rec = pd.read_csv(run / "recovery.csv")
        rec = rec[rec.inferred]
        for r in rec.itertuples():
            rows.append({
                "gt": gt, "n": int(n), "seed": int(s), "parameter": r.parameter, "truth": r.truth,
                "mean": r.posterior_mean, "sd": r.posterior_std, "covered": bool(r.covered),
                "z": r.z_score, "rhat": e["max_rhat"], "chains": e["chains"],
                "stop_reason": stop.get("reason"), "stop_met": stop.get("met"),
                "draws": e.get("draws_per_chain"), "hours": met["total_seconds"] / 3600,
                "grad_evals": e["gradient_evals"],
                "chain_means_BtoA": e["chain_means"].get("B.to_A"),
            })  # fmt: skip
    return pd.DataFrame(rows)


def mle_values(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.glob("*/n*_s*/mle.json")):
        m = json.loads(p.read_text())
        gt = p.parent.parent.name
        n, s = p.parent.name[1:].split("_s")
        for q in m["parameters"]:
            rows.append({"gt": gt, "n": int(n), "seed": int(s), "parameter": q, "mle": m["mle"][q]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ large n
def large_n(xg2: pd.DataFrame, mle: pd.DataFrame):
    b = xg2[(xg2.parameter == "B.to_A") & xg2["gt"].isin(CUR)]
    bm = mle[(mle.parameter == "B.to_A") & mle["gt"].isin(CUR)]
    ns = sorted(b.n.unique())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for ax, gt in zip(axes.flat, CUR, strict=True):
        g, gm = b[b["gt"] == gt], bm[bm["gt"] == gt]
        for s, off in zip((0, 1, 2), (-0.09, 0, 0.09), strict=True):
            gs = g[g.seed == s].sort_values("n")
            ax.errorbar(np.log2(gs.n) + off, gs["mean"], yerr=gs.sd, fmt="o", color=BLUE, ms=3.2,
                        lw=0.9, label="posterior mean ± sd" if s == 0 else None)  # fmt: skip
            gms = gm[gm.seed == s].sort_values("n")
            ax.plot(np.log2(gms.n) + off + 0.04, gms.mle, "D", mfc="none", mec=ORANGE, ms=3.6,
                    mew=0.9, label="MLE" if s == 0 else None)  # fmt: skip
        ax.axhline(g.truth.iloc[0], color=INK, lw=0.9, ls="--", label="truth")
        ax.set_title(LABEL[gt])
        ax.set_xticks(np.log2(ns), [str(n) for n in ns])
        ax.set_ylim(-0.03, 1.0)
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("B$\\to$A")
    axes[0, 0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    mf.save(fig, "btoa_large_n")
    rows = []
    for gt in CUR:
        for n in (1024, 2048, 4096):
            g = b[(b["gt"] == gt) & (b.n == n)]
            gm = bm[(bm["gt"] == gt) & (bm.n == n)]
            t = g.truth.iloc[0]
            rows.append({
                "truth": LABEL[gt], "n": str(n), "btoa": f"{t:.2f}",
                "post": f"{g['mean'].mean():.3f} ({g.sd.mean():.3f})", "z": f"{g.z.mean():.2f}",
                "cov": f"{int(g.covered.sum())}/3",
                "rmse_post": f"{np.sqrt(np.mean((g['mean'] - t) ** 2)):.3f}",
                "rmse_mle": f"{np.sqrt(np.mean((gm.mle - t) ** 2)):.3f}",
                "rhat": f"{g.rhat.max():.3f}",
            })  # fmt: skip
    mf.tex_table(pd.DataFrame(rows), "large_n")


# ------------------------------------------------------------------ flat B->A prior
def flat_btoa(xg2: pd.DataFrame, fb: pd.DataFrame):
    a = xg2[xg2.n <= 1024]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0), sharex=True)
    for ax, gt in zip(axes.flat, GTS, strict=True):
        for df, col, mk, lab, off in ((a, BLUE, "o", "uniform Dirichlet (XG2)", -0.06),
                                      (fb, ORANGE, "D", "flat B$\\to$A prior", 0.06)):  # fmt: skip
            g = df[(df["gt"] == gt) & (df.parameter == "B.to_A")].groupby("n").agg(
                m=("mean", "mean"), s=("sd", "mean"))
            ax.errorbar(np.log2(g.index) + off, g.m, yerr=g.s, fmt=mk + "-", color=col, ms=3.2,
                        lw=0.9, label=lab)  # fmt: skip
        t = a[(a["gt"] == gt) & (a.parameter == "B.to_A")].truth.iloc[0]
        ax.axhline(t, color=INK, lw=0.9, ls="--")
        ax.set_title(LABEL[gt])
        ax.set_ylim(-0.03, 1.0)
        ax.set_xticks(np.log2(mf.NS), [str(n) for n in mf.NS])
    for ax in axes[-1]:
        ax.set_xlabel("clones per endpoint (n)")
    for ax in axes[:, 0]:
        ax.set_ylabel("B$\\to$A (mean over datasets)")
    axes[0, 0].legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    mf.save(fig, "flat_btoa")
    rows = []
    for gt in GTS:
        r = {"truth": LABEL[gt]}
        for n in (128, 1024):
            for key, df in (("u", a), ("f", fb)):
                g = df[(df["gt"] == gt) & (df.n == n)]
                b = g[g.parameter == "B.to_A"]
                t = b.truth.iloc[0]
                r[f"b{key}{n}"] = np.sqrt(np.mean((b["mean"] - t) ** 2))
                r[f"a{key}{n}"] = np.sqrt(np.mean((g["mean"] - g.truth) ** 2))
                inner = g[g.truth > 0]
                r[f"c{key}{n}"] = inner.covered.mean()
        rows.append(r)
    d = pd.DataFrame(rows)
    out = pd.DataFrame({"truth": d.truth})
    for n in (128, 1024):
        out[f"b{n}"] = [(f"{u:.2f} / \\textbf{{{f:.2f}}}" if f < u else f"\\textbf{{{u:.2f}}} / {f:.2f}")
                        for u, f in zip(d[f"bu{n}"], d[f"bf{n}"], strict=True)]  # fmt: skip
        out[f"a{n}"] = [(f"{u:.3f} / \\textbf{{{f:.3f}}}" if f < u else f"\\textbf{{{u:.3f}}} / {f:.3f}")
                        for u, f in zip(d[f"au{n}"], d[f"af{n}"], strict=True)]  # fmt: skip
        out[f"c{n}"] = [f"{100 * u:.0f} / {100 * f:.0f}\\%"
                        for u, f in zip(d[f"cu{n}"], d[f"cf{n}"], strict=True)]  # fmt: skip
    mf.tex_table(out[["truth", "b128", "a128", "c128", "b1024", "a1024", "c1024"]], "flat_btoa")
    runs = fb.groupby(["gt", "n", "seed"]).first().reset_index()
    stop = runs.groupby("gt").agg(runs=("seed", "size"), met=("stop_met", "sum"),
                                  hours=("hours", "median"))  # fmt: skip
    stop = stop.reindex(GTS)
    mf.tex_table(pd.DataFrame({
        "truth": [LABEL[g] for g in stop.index], "met": [f"{int(m)}/{int(r)}" for m, r in zip(stop.met, stop.runs, strict=True)],
        "hours": [f"{h:.2f}" for h in stop.hours]}), "flat_btoa_stop")  # fmt: skip
    summary = {"flat_btoa_runs": int(len(runs)), "stop_met": int(runs.stop_met.sum()),
               "max_draws_cells": runs[runs.stop_met == False][["gt", "n", "seed"]].values.tolist()}  # noqa: E712
    return summary


# ------------------------------------------------------------------ reruns and noise
def bimodal(xg2: pd.DataFrame, rr: pd.DataFrame):
    rows = []
    for (gt, n, s), g in rr.groupby(["gt", "n", "seed"]):
        o = xg2[(xg2["gt"] == gt) & (xg2.n == n) & (xg2.seed == s)]
        ob, rb = o[o.parameter == "B.to_A"].iloc[0], g[g.parameter == "B.to_A"].iloc[0]
        cm = np.array(rb.chains_means if hasattr(rb, "chains_means") else rb.chain_means_BtoA)
        rows.append({
            "truth": LABEL[gt], "n": str(n), "seed": str(s), "rhat_o": f"{ob.rhat:.2f}",
            "post_o": f"{ob['mean']:.2f} ({ob.sd:.2f})", "rhat_r": f"{rb.rhat:.2f}",
            "post_r": f"{rb['mean']:.2f} ({rb.sd:.2f})",
            "chains": f"{cm.min():.2f}--{cm.max():.2f}", "t": f"{ob.truth:.2f}",
        })  # fmt: skip
    mf.tex_table(pd.DataFrame(rows), "bimodal")


def sampler_noise(xg2: pd.DataFrame, nz: pd.DataFrame):
    rows = []
    for gt in ["T1_ref", "T3_Bdiv", "P3"]:
        samp = pd.concat([xg2[(xg2["gt"] == gt) & (xg2.n == 1024) & (xg2.seed == 0)],
                          nz[(nz["gt"] == gt) & (nz.n == 1024)]])  # fmt: skip
        data = xg2[(xg2["gt"] == gt) & (xg2.n == 1024)]
        for p in ("B.to_A", "A.div_AA", "B.div_BB", "A.div_AB"):
            a, b = samp[samp.parameter == p], data[data.parameter == p]
            rows.append({"truth": LABEL[gt], "parameter": mf.pname(p),
                         "sampler": f"{a['mean'].std(ddof=1):.3f}", "data": f"{b['mean'].std(ddof=1):.3f}",
                         "sd": f"{a.sd.mean():.3f}",
                         "ratio": f"{a['mean'].std(ddof=1) / a.sd.mean():.2f}"})  # fmt: skip
    mf.tex_table(pd.DataFrame(rows), "sampler_noise")


def reparam(rb: Path):
    d = pd.read_csv(rb / "runs.csv")
    keep = d[d.arm.isin(["baseline", "brow_probit", "probit", "brow", "baseline_seed1",
                         "brow_probit_seed1", "smc", "smc_brow_probit"])]  # fmt: skip
    rows = []
    for r in keep.itertuples():
        rows.append({"cell": r.cell.replace("_n", " $n$=").replace("_s", " s").replace("_", " "),
                     "arm": r.arm.replace("_", "+").replace("+seed1", " (seed 1)"),
                     "evals": f"{r.grad_evals / 1e3:.0f}k", "ess": f"{r.ess_bulk:.0f} / {r.ess_tail:.0f}",
                     "per1k": f"{r.bulk_per_1k_grad:.2f}", "rhat": f"{r.max_rank_rhat:.3f}",
                     "div": f"{r.divergences:.0f}",
                     "low": "" if pd.isna(r.low_branch_pooled) else f"{100 * r.low_branch_pooled:.0f}\\%"})  # fmt: skip
    mf.tex_table(pd.DataFrame(rows), "reparam")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=mf.ROOT / "results")
    E = ap.parse_args().results / "exact_gw"
    xg2 = load_root(E / "data_size")
    mle = mle_values(E / "data_size")
    large_n(xg2, mle)
    fb = load_root(E / "data_size_flat_btoa")
    s = flat_btoa(xg2, fb)
    bimodal(xg2, load_root(E / "data_size_reruns" / "bimodal"))
    sampler_noise(xg2, load_root(E / "data_size_reruns" / "sampler_noise"))
    reparam(E / "reparam_bench")
    (mf.TAB / "followups_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, default=str)[:600])


if __name__ == "__main__":
    main()
