"""X3b: should a Galton-Watson model be fitted to Bellman-Harris data?

Arms on identical BH datasets (scenario x design x seed; data_seed = seed):
exact BH (well specified, ``results/x3b/bh_exact``), NPE BH (``npe_bh``), exact GW with
tau in {20, 24, 25, 30} h (``gw_exact``), exact GW with tau chosen by evidence
(``gw_tau/<scenario>/<design>/seed<k>/best.json``, scripts/x3b_tau_select.py) and the
held-out arms (``*_holdout``: fit on 80 % of the organoids, test lpd on 20 %).
Directory layout: ``results/x3b/<arm>/<scenario>/<design>/<job>/``.

Outputs (``--out``, default ``results/x3b/analysis``):

* ``x3b_runs.csv`` -- one row per run: arm, scenario, design, seed, dataset digest, MAE /
  bias / 90 % coverage on the shared probabilities, PPC, log evidence (+- MC se), held-out
  lpd, posterior mean / sd of the derived quantities and their truths;
* ``x3b_parameters.csv`` -- run x parameter (shared probabilities + derived quantities):
  truth, posterior mean, sd, 5 / 95 % quantiles, covered, error;
* ``x3b_groups.csv`` -- scenario x design x arm: bias, MAE, coverage, sd, per quantity;
* ``x3b_detectability.csv`` -- scenario x design x GW arm, paired by dataset with the exact
  BH run: log Bayes factor BH vs GW (mean, min, share > log 100), held-out delta lpd
  (BH - GW, from the holdout arms), PPC of both;
* ``x3b_dataset_check.csv`` -- dataset digests per (scenario, design, seed) and arm; every
  arm must share one digest (``--check-datasets`` prints mismatches and exits 1 if any);
* ``figures/x3b_consistency_<scenario>.png`` -- posterior mean +- sd vs design, BH vs GW,
  for A self-renewal, B.to_A, growth rate per hour, stable B:A ratio.

Usage: ``uv run python scripts/analyse_x3b.py [--root results/x3b] [--check-datasets]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bio_sbi.datacard import job_record
from bio_sbi.derived import COLUMNS as DERIVED
from bio_sbi.derived import derived_quantities

ARMS = {"bh_exact": "BH exact", "npe_bh": "BH NPE", "gw_exact": "GW",
        "bh_exact_holdout": "BH exact (holdout)", "gw_exact_holdout": "GW (holdout)"}  # fmt: skip
DESIGN_ORDER = ["real_like", "d512", "d2048"]
DERIVED_SHOWN = ["A_self_renewal", "growth_rate_per_h", "stable_B_to_A"]
LEVEL = 0.9


def _sim(cfg, section):
    return instantiate(section.model, _convert_="all")


def load_run(job: Path, arm: str, scenario: str, design: str) -> tuple[dict, list[dict]] | None:
    """Run row and parameter rows of one finished job (``None`` if unfinished)."""
    mfile = job / "metrics.json"
    if not mfile.exists():
        return None
    m = json.loads(mfile.read_text())
    cfg = OmegaConf.load(job / ".hydra" / "config.yaml")
    sim = _sim(cfg, cfg.simulator)
    data_sim = _sim(cfg, cfg.data.simulator)
    ref = np.load(job / "reference.npz")
    truth = dict(zip([str(n) for n in ref["truth_names"]], ref["truth"].tolist(), strict=True))
    samples = pd.read_csv(job / "posterior_samples.csv.gz")
    tau = cfg.simulator.model.get("hours_per_generation") if "galton" in cfg.simulator.name else None
    label = ARMS[arm] + (f" tau={float(tau):g}" if tau is not None else "")
    shared = [r["parameter"] for r in m["recovery"]["per_parameter"]
              if r["inferred"] and "lifetime" not in r["parameter"]]  # fmt: skip
    q = (1 - LEVEL) / 2
    pars = []
    for p in shared:
        x = samples[p].to_numpy()
        t = truth.get(p, np.nan)
        lo, hi = np.quantile(x, [q, 1 - q])
        pars.append({"quantity": p, "kind": "probability", "truth": t, "mean": x.mean(),
                     "sd": x.std(), "q05": lo, "q95": hi, "covered": bool(lo <= t <= hi),
                     "error": x.mean() - t})  # fmt: skip
    d_post = derived_quantities(sim, samples[list(sim.parameter_names)].to_numpy())
    d_true = derived_quantities(data_sim, ref["truth"][None]).iloc[0]
    for c in DERIVED:
        x = d_post[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        t = float(d_true[c])
        lo, hi = np.quantile(x, [q, 1 - q]) if len(x) else (np.nan, np.nan)
        pars.append({"quantity": c, "kind": "derived", "truth": t, "mean": np.mean(x),
                     "sd": np.std(x), "q05": lo, "q95": hi, "covered": bool(lo <= t <= hi),
                     "error": np.mean(x) - t})  # fmt: skip
    for c in ("A.mean_lifetime", "B.mean_lifetime"):
        if c in samples and sim.space.free_names and c in sim.space.free_names:
            x = samples[c].to_numpy()
            lo, hi = np.quantile(x, [q, 1 - q])
            t = truth.get(c, np.nan)
            pars.append({"quantity": c, "kind": "lifetime", "truth": t, "mean": x.mean(),
                         "sd": x.std(), "q05": lo, "q95": hi, "covered": bool(lo <= t <= hi),
                         "error": x.mean() - t})  # fmt: skip
    rec = job_record(job)
    method = m.get("exact_bh") or m.get("exact_gw") or {}
    prob = [r for r in pars if r["kind"] == "probability"]
    ho = m.get("heldout") or {}
    row = {
        "arm": label, "family": "GW" if tau is not None else "BH", "source": arm,
        "tau": float(tau) if tau is not None else np.nan, "scenario": scenario,
        "design": design, "seed": int(m["seed"]), "data_seed": int(m["data_seed"]),
        "dataset": rec.get("dataset"), "job": str(job),
        "mae": np.mean([abs(r["error"]) for r in prob]),
        "bias": np.mean([r["error"] for r in prob]),
        "coverage90": np.mean([r["covered"] for r in prob if r["truth"] not in (0.0, 1.0)]),
        "mean_sd": np.mean([r["sd"] for r in prob]),
        "ppc_out95": (m.get("ppc") or {}).get("fraction_outside_95", np.nan),
        "log_evidence": m.get("log_evidence", np.nan),
        "log_evidence_se": m.get("log_evidence_se", np.nan),
        "heldout_lpd": ho.get("lpd", np.nan), "heldout_lpd_se": ho.get("lpd_se", np.nan),
        "heldout_n_test": ho.get("n_test", np.nan),
        "reliable": method.get("reliable", np.nan),
        "max_rhat": method.get("max_rhat", (method.get("smc") or {}).get("max_nested_rhat")),
        "ess": method.get("ess", method.get("min_ess")),
        "seconds": m.get("total_seconds"),
        **{f"{r['quantity']}_mean": r["mean"] for r in pars if r["kind"] != "probability"},
        **{f"{r['quantity']}_sd": r["sd"] for r in pars if r["kind"] != "probability"},
        **{f"{r['quantity']}_truth": r["truth"] for r in pars if r["kind"] != "probability"},
    }  # fmt: skip
    base = {k: row[k] for k in ("arm", "family", "scenario", "design", "seed", "dataset")}
    return row, [{**base, **r} for r in pars]


def tau_selected(root: Path) -> list[dict]:
    """Rows for the GW arm with tau chosen by evidence (gw_tau/<scen>/<design>/seed<k>)."""
    out = []
    for best in sorted(root.glob("gw_tau/*/*/*/best.json")):
        b = json.loads(best.read_text())
        if not b.get("run"):
            continue
        scen, design = best.parent.parent.parent.name, best.parent.parent.name
        r = load_run(Path(b["run"]), "gw_exact", scen, design)
        if r is None:
            continue
        row, pars = r
        row["arm"] = "GW tau=evidence"
        row["tau"] = b["tau"]
        row["tau_generations"] = b.get("generations")
        out.append((row, [{**p, "arm": "GW tau=evidence"} for p in pars]))
    return out


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pars = [], []
    for arm in ARMS:
        for job in sorted(root.glob(f"{arm}/*/*/*/")):
            if not (job / ".hydra").exists():
                continue
            scen, design = job.parent.parent.name, job.parent.name
            try:
                r = load_run(job, arm, scen, design)
            except Exception as err:  # noqa: BLE001 - report and continue
                print(f"skip {job}: {err}", file=sys.stderr)
                continue
            if r:
                rows.append(r[0])
                pars += r[1]
    for row, p in tau_selected(root):
        rows.append(row)
        pars += p
    return pd.DataFrame(rows), pd.DataFrame(pars)


def groups(pars: pd.DataFrame) -> pd.DataFrame:
    g = pars.groupby(["scenario", "design", "arm", "quantity"])
    out = g.agg(runs=("seed", "nunique"), truth=("truth", "first"), mean=("mean", "mean"),
                sd=("sd", "mean"), bias=("error", "mean"),
                mae=("error", lambda e: np.mean(np.abs(e))),
                coverage90=("covered", "mean"))  # fmt: skip
    return out.reset_index()


def detectability(runs: pd.DataFrame) -> pd.DataFrame:
    """Paired by dataset: BH exact vs every GW arm (evidence; holdout lpd from holdout arms)."""
    rows = []
    main = runs[~runs.source.str.endswith("holdout")]
    hold = runs[runs.source.str.endswith("holdout")]
    bh = main[main.arm == "BH exact"].set_index(["scenario", "design", "seed"])
    bh_h = hold[hold.family == "BH"].set_index(["scenario", "design", "seed"])
    for arm, gw in main[main.family == "GW"].groupby("arm"):
        gw = gw.set_index(["scenario", "design", "seed"])
        both = bh.join(gw, lsuffix="_bh", rsuffix="_gw", how="inner")
        if both.empty:
            continue
        both["log_bf"] = both.log_evidence_bh - both.log_evidence_gw
        both["log_bf_se"] = np.hypot(both.log_evidence_se_bh, both.log_evidence_se_gw)
        both["same_data"] = both.dataset_bh == both.dataset_gw
        gw_h = hold[(hold.family == "GW") & (hold.tau == gw.tau.iloc[0])].set_index(
            ["scenario", "design", "seed"])  # fmt: skip
        lp = bh_h[["heldout_lpd", "heldout_lpd_se"]].join(
            gw_h[["heldout_lpd", "heldout_lpd_se"]], lsuffix="_bh", rsuffix="_gw", how="inner")
        lp["dlpd"] = lp.heldout_lpd_bh - lp.heldout_lpd_gw
        for (scen, design), b in both.groupby(level=[0, 1]):
            l_ = lp.loc[(scen, design)] if (scen, design) in lp.index.droplevel(2) else None
            rows.append({
                "scenario": scen, "design": design, "gw_arm": arm, "datasets": len(b),
                "all_same_data": bool(b.same_data.all()),
                "log_bf_mean": b.log_bf.mean(), "log_bf_min": b.log_bf.min(),
                "log_bf_max": b.log_bf.max(), "log_bf_se_max": b.log_bf_se.max(),
                "share_bf_gt_100": float((b.log_bf > np.log(100)).mean()),
                "share_bf_favours_gw": float((b.log_bf < 0).mean()),
                "dlpd_mean": l_.dlpd.mean() if l_ is not None else np.nan,
                "dlpd_min": l_.dlpd.min() if l_ is not None else np.nan,
                "holdout_pairs": len(l_) if l_ is not None else 0,
                "ppc_bh": b.ppc_out95_bh.mean(), "ppc_gw": b.ppc_out95_gw.mean(),
                "mae_bh": b.mae_bh.mean(), "mae_gw": b.mae_gw.mean(),
            })  # fmt: skip
    return pd.DataFrame(rows)


def dataset_check(runs: pd.DataFrame) -> pd.DataFrame:
    t = runs.pivot_table(index=["scenario", "design", "seed"], columns="arm", values="dataset",
                         aggfunc="first")  # fmt: skip
    t["consistent"] = t.apply(lambda r: r.dropna().nunique() <= 1, axis=1)
    return t.reset_index()


def consistency_figures(pars: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ["A.div_AA", "B.to_A", "growth_rate_per_h", "stable_B_to_A"]
    arms = ["BH exact", "BH NPE", "GW tau=20", "GW tau=24", "GW tau=30", "GW tau=evidence"]
    colors = dict(zip(arms, ["#1b6ca8", "#7fb3d5", "#e8a33d", "#c0392b", "#8e5a2b", "#6c3483"],
                      strict=True))  # fmt: skip
    for scen, s in pars.groupby("scenario"):
        designs = [d for d in DESIGN_ORDER if d in set(s.design)]
        fig, axes = plt.subplots(1, len(shown), figsize=(4 * len(shown), 3.4))
        for ax, qn in zip(axes, shown, strict=True):
            q = s[s.quantity == qn]
            if q.empty:
                ax.set_visible(False)
                continue
            for i, arm in enumerate([a for a in arms if a in set(q.arm)]):
                a = q[q.arm == arm].groupby("design").agg(mean=("mean", "mean"),
                                                          sd=("sd", "mean"))  # fmt: skip
                x = [designs.index(d) + 0.08 * (i - 2.5) for d in a.index if d in designs]
                a = a.loc[[d for d in a.index if d in designs]]
                ax.errorbar(x, a["mean"], yerr=a["sd"], fmt="o-", ms=4, lw=1.2, capsize=2,
                            color=colors[arm], label=arm)  # fmt: skip
            ax.axhline(q.truth.iloc[0], color="0.3", ls="--", lw=1)
            ax.set_xticks(range(len(designs)), designs)
            ax.set_title(qn, fontsize=10)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"X3b {scen}: posterior mean +- sd (mean over seeds) vs design; dashed = truth",
                     fontsize=10)  # fmt: skip
        fig.tight_layout()
        fig.savefig(out / f"x3b_consistency_{scen}.png", dpi=130)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/x3b"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--check-datasets", action="store_true")
    args = p.parse_args(argv)
    out = args.out or args.root / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs, pars = collect(args.root)
    if runs.empty:
        print(f"no finished runs under {args.root}")
        return
    check = dataset_check(runs)
    check.to_csv(out / "x3b_dataset_check.csv", index=False)
    bad = check[~check.consistent]
    if len(bad):
        print("DATASET MISMATCH (arms of one scenario/design/seed differ):")
        print(bad.to_string(index=False))
    if args.check_datasets:
        print(f"{len(check)} datasets, {len(bad)} inconsistent")
        sys.exit(1 if len(bad) else 0)
    runs.to_csv(out / "x3b_runs.csv", index=False)
    pars.to_csv(out / "x3b_parameters.csv", index=False)
    g = groups(pars)
    g.to_csv(out / "x3b_groups.csv", index=False)
    det = detectability(runs)
    det.to_csv(out / "x3b_detectability.csv", index=False)
    consistency_figures(pars, out / "figures")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["scenario", "design", "arm", "seed", "mae", "coverage90", "ppc_out95",
            "log_evidence", "log_evidence_se", "heldout_lpd", "growth_rate_per_h_mean",
            "stable_B_to_A_mean"]  # fmt: skip
    print(runs[[c for c in cols if c in runs]].round(4).to_string(index=False))
    key = g[g.quantity.isin(["A.div_AA", "B.to_A", "B.persist", *DERIVED_SHOWN])]
    print(key.round(4).to_string(index=False))
    if not det.empty:
        print(det.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
